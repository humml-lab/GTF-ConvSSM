"""
    loss(tfrec, X̃, S̃, ̃R)

Performs a forward pass using the teacher forced recursion wrapper `tfrec` and
computes and return the loss w.r.t. data `X̃`. Optionally external inputs `S̃` as 
well as nuisance artifacts can be provided.
"""

function loss(
    d::ExternalInputsNuisanceArtifactsDataset,
    tfrec::AbstractTFRecur,
    X̃::AbstractArray{T, 3},
    S̃::AbstractArray{T, 3},
    R̃::AbstractArray{T, 3},
) where {T}
    Z = tfrec(d, X̃, S̃, R̃)
    X̂ = tfrec.O(Z, R̃[:, :, 2:end])
    return @views Flux.mse(X̂, X̃[:, :, 2:end])
end

function train_!(
    m::AbstractPLRNN,
    O::ObservationModel,
    d::ExternalInputsNuisanceArtifactsDataset,
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

    for e = 1:E
        # process a couple of batches
        t₁ = time_ns()
        for sₑ = 1:Sₑ
            # sample a batch
            X̃, S̃, R̃ = sample_batch(d, T̃, S)

            # add noise noise if noise level > 0
            σ_noise > zero(σ_noise) ? add_gaussian_noise!(X̃, σ_noise) : nothing

            # forward and backward pass
            grads = Flux.gradient(θ) do
                # compute loss
                Lₜᵣ = loss(d, tfrec, X̃, S̃, R̃)
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

        # abort training if NaNs present in θ
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
            X̃, S̃, R̃ = sample_batch(d, T̃, S)
            Lₜᵣ = loss(d, tfrec, X̃, S̃, R̃)
            Lᵣ = regularization_loss(tfrec, κ, λₘₐᵣ, λₗ, λₒ)

            # generated trajectory
            X_gen = @allowscalar @views generate(d, tfrec.model, tfrec.O, d.X[1, :], vcat(d.S, d.S_test), vcat(d.R, d.R_test), T_total)

            # move data to cpu for metrics and plotting
            X_cpu = vcat(d.X, d.X_test) |> cpu
            X_gen_cpu = X_gen |> cpu

            # metrics
            D_stsp = state_space_distance(X_cpu, X_gen_cpu, scal)
            pse, _ = power_spectrum_error(X_cpu, X_gen_cpu, σ_smoothing)
            pe = prediction_error(d, tfrec.model, tfrec.O, d.X, d.S, d.R, PE_n)
            pe_test = isempty(d.X_test) ? NaN : prediction_error(d, tfrec.model, tfrec.O, d.X_test, d.S_test, d.R_test, PE_n)
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
                )
            end
        end
    end
    #save the loss metrics as csv file and plot it
    plot_loss(prog_metrics, save_path)
end
