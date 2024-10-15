"""
    loss(tfrec, X̃, S̃)

Performs a forward pass using the teacher forced recursion wrapper `tfrec` and
computes and return the loss w.r.t. data `X̃`. Optionally external inputs `S̃`
can be provided.
"""
function loss(
    d::ExternalInputsDatasetConv,
    tfrec::AbstractTFRecur,
    X̃::AbstractArray{T, 3},
    S̃::AbstractArray{T, 3},
    X̃_deconv::AbstractArray{T, 3},
    hrf::AbstractVector{T},
    conv_mat::AbstractMatrix{T} #Flux can not handle DSP.conv so convolution with matrix
    ) where {T}
    Z = tfrec(d, X̃_deconv, S̃)
    M, S, T̃ = size(Z)
    Z_conv = reshape((conv_mat*(reshape(Z, :, T̃))')', M, S, :)
    X̂ = tfrec.O(Z_conv) 
    return @views Flux.mse(X̂, X̃[:, :, 1+length(hrf):end])
end


function train_!(
    m::AbstractPLRNN,
    O::ObservationModel,
    d::ExternalInputsDatasetConv,
    opt::Flux.Optimise.Optimiser,
    args::AbstractDict,
    save_path::String,
)
    # data shape
    T, N = size(d.X)
    T_total = size(d.X, 1) + size(d.X_test, 1)

    # hypers (type hinting reduces startup time drastically)
    E = args["epochs"]::Int
    M = args["latent_dim"]::Int
    Sₑ = args["batches_per_epoch"]::Int
    S = args["batch_size"]::Int
    τ = args["teacher_forcing_interval"]::Int
    σ_noise = args["gaussian_noise_level"]::Float32
    T̃ = args["sequence_length"]::Int
    κ = args["MAR_ratio"]::Float32
    λₘₐᵣ = args["MAR_lambda"]::Float32
    σ²_scaling = args["D_stsp_scaling"]::Float32
    bins = args["D_stsp_bins"]::Int
    σ_smoothing = args["PSE_smoothing"]::Float32
    PE_n = args["PE_n"]::Int
    isi = args["image_saving_interval"]::Int
    ssi = args["scalar_saving_interval"]::Int
    exp = args["experiment"]::String
    name = args["name"]::String
    run = args["run"]::Int
    α = args["weak_tf_alpha"]::Float32
    λₒ = args["obs_model_regularization"]::Float32
    λₗ = args["lat_model_regularization"]::Float32

    prog = Progress(joinpath(exp, name), run, 20, E, 0.8)
    stop_flag = false

    # decide on D_stsp scaling
    scal, stsp_name = decide_on_measure(σ²_scaling, bins, N)

    # initialize stateful model wrapper
    tfrec = choose_recur_wrapper(m, d, O, M, N, S, τ, α)

    # model parameters
    θ = Flux.params(tfrec)

    #initialize DataFrame to save the progress metrics
    prog_metrics = DataFrame(Epoch = Int[], Loss = Float32[], PE_train = Float32[], PE_test = Float32[], D_stsp = Float32[], PSE = Float32[])

    #define quantities to simplify expressions, e.g. in the prediction errors
    cutl = findfirst(!isnan,d.X_deconv[:,1])-1
    cutr = findfirst(!isnan,d.X_deconv[end:-1:1,1])-1

    #convolutionMatrix to determine the MSE while training (fitting the size of the sequences)
    conv_mat = sparse(getConvolutionMatrix(d.hrf, T̃))
    
    #plot deconv data
    plot_reconstruction(
        d.X_deconv,
        d.X,
        joinpath(save_path, "Data_and_Deconv.png");
        X̃_test = d.X_deconv_test,
        X_test = d.X_test,
        cutr = cutr,
    )

    for e = 1:E
        # process a couple of batches
        t₁ = time_ns()
        for sₑ = 1:Sₑ
            # sample a batch
            X̃, S̃, X̃_deconv = sample_batch(d, T̃, S)

            # add noise noise if noise level > 0
            σ_noise > zero(σ_noise) ? (add_gaussian_noise!(X̃, σ_noise), add_gaussian_noise!(X̃_deconv, σ_noise)) : nothing

            # forward and backward pass
            grads = Flux.gradient(θ) do
                Lₜᵣ = loss(d, tfrec, X̃, S̃, X̃_deconv, Float32.(d.hrf), Float32.(conv_mat))
                Lᵣ = regularization_loss(tfrec, κ, λₘₐᵣ, λₗ, λₒ)
                return Lₜᵣ + Lᵣ
            end

            # keep W matrix offdiagonal by setting gradients to zero
            keep_connectivity_offdiagonal!(tfrec.model, grads)

            # optimiser step
            Flux.Optimise.update!(opt, θ, grads)

            # check for NaNs in parameters (exploding gradients)
            stop_flag = check_for_NaNs(θ)
            if stop_flag
                break
            end
        end
        if stop_flag
            save_model(
                [tfrec.model, tfrec.O],
                joinpath(save_path, "checkpoints", "model_$e.bson"),
            )
            @warn "NaN(s) in parameters detected! \
                This is likely due to exploding gradients. Aborting training..."
            break
        end
        t₂ = time_ns()
        Δt = (t₂ - t₁) / 1e9
        update!(prog, Δt, e)

        # plot trajectory
        if e % ssi == 0
            # loss
            X̃, S̃, X̃_deconv = sample_batch(d, T̃, S)
            Lₜᵣ = loss(d, tfrec, X̃, S̃, X̃_deconv, d.hrf, conv_mat)
            Lᵣ = regularization_loss(tfrec, κ, λₘₐᵣ, λₗ, λₒ)

            # generated trajectory
            X_gen = @allowscalar @views generate(d, tfrec.model, tfrec.O, d.X_deconv[1+cutl, :], vcat(d.S[1+cutl:end, :], d.S_test), T_total - cutl)

            # move data to cpu for metrics and plotting
            X_cpu = vcat(d.X, d.X_test)[1:size(X_gen, 1),:] |> cpu
            X_gen_cpu = X_gen |> cpu

            # metrics
            D_stsp = state_space_distance(X_cpu, X_gen_cpu, scal)
            pse, _ = power_spectrum_error(X_cpu, X_gen_cpu, σ_smoothing)
            pe, pe_test = prediction_error(d, tfrec.model, tfrec.O, PE_n)
            # progress printing
            scalars = gather_scalars(Lₜᵣ, Lᵣ, D_stsp, stsp_name, pse, pe, pe_test, PE_n)
            print_progress(prog, Δt, scalars)

            #Add progress metrics to DataFrame
            push!(prog_metrics, Dict(:Epoch => e, :Loss => Lₜᵣ, :PE_train => pe, :PE_test => pe_test, :D_stsp => D_stsp, :PSE => pse))

            save_model(
                [tfrec.model, tfrec.O],
                joinpath(save_path, "checkpoints", "model_$e.bson"),
            )
            if e % isi == 0
                # plot
                plot_reconstruction(
                    X_gen_cpu,
                    X_cpu[1:T, :],
                    joinpath(save_path, "plots", "generated_$e.png");
                    X_test = X_cpu[T+1:end,:],
                    cutl = cutl,
                )
            end
        end
    end
    #save the loss metrics as csv file and plot it
    plot_loss(prog_metrics, save_path)
end
