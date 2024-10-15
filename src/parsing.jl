using ArgParse
using Flux

function initialize_model(args::AbstractDict, D::AbstractDataset; mod = @__MODULE__)
    # gather args
    M = args["latent_dim"]
    id_tf = args["observation_model"] == "Identity"

    B = args["num_bases"]
    Layers = args["hidden_layers"]
    model_name = args["model"]
    hidden_dim = args["hidden_dim"]

    # model type in correct module scope
    model_t = @eval mod $(Symbol(model_name))

    # specify model args based on model type
    if model_t <: AbstractVanillaPLRNN
        model_args = (M,)
    elseif model_t <: AbstractDendriticPLRNN
        model_args = id_tf ? (M, B, D.X) : (M, B)
    elseif model_t <: AbstractDeepPLRNN
        model_args = (M, Layers)
    elseif model_t <: AbstractShallowPLRNN
        model_args = (M, hidden_dim)
    end

    # external inputs?
    K = !isempty(args["path_to_inputs"]) ? (size(D.S, 2),) : ()

    # optional arguments
    opt_args = args["optional_model_args"]

    # initialize model
    model = model_t(model_args..., opt_args..., K...)

    println("Model / # Parameters: $(typeof(model)) / $(num_params(model))")
    return model
end

function initialize_observation_model(args::AbstractDict, D::AbstractDataset)
    N = size(D.X, 2)
    M = args["latent_dim"]

    if !isdefined(D,:R) #dataset has no nuisance artifacts (which would be relevant for the observation model)
        # initialize by default w/o bias
        if args["observation_model"] == "Affine"
            obs_model = Affine(N, M; use_bias = false)
        elseif args["observation_model"] == "Identity"
            obs_model = Identity(N, M)
        elseif args["observation_model"] == "Regressor"
            error("No nuisance artifacts given: Regressor observation model cannot be used. 
            Choose a different observation model.")
        end
    else
        P = size(D.R, 2) #dimension of the nuisance artifacts
        if args["observation_model"] == "Regressor"
            obs_model = Regressor(N, M, P)
        else
            error("Nuisance artifacts given: This observation model does not take them into account.
            Choose a different observation model.")
        end
    end

    println("Obs. Model / # Parameters: $(typeof(obs_model)) / $(num_params(obs_model))")
    return obs_model
end

function initialize_optimizer(args::Dict{String, Any})
    # optimizer chain
    opt_vec = []

    # vars
    κ = args["gradient_clipping_norm"]::Float32
    ηₛ = args["start_lr"]::Float32
    ηₑ = args["end_lr"]::Float32
    E = args["epochs"]::Int
    bpe = args["batches_per_epoch"]::Int

    # set gradient clipping
    if κ > zero(κ)
        push!(opt_vec, ClipNorm(κ))
    end

    # set SGD optimzier (ADAM, RAdam, etc)
    opt_sym = Symbol(args["optimizer"])
    opt = @eval $opt_sym($ηₛ)
    push!(opt_vec, opt)

    # set exponential decay learning rate scheduler
    γ = exp(log(ηₑ / ηₛ) / E)
    decay = ExpDecay(1, γ, bpe, ηₑ, 1)
    push!(opt_vec, decay)

    return Flux.Optimise.Optimiser(opt_vec...)
end

get_device(args::AbstractDict) =
    if args["device"] == "gpu"
        return gpu
    else
        return cpu
    end

"""
    argtable()

Prepare the argument table holding the information of all possible arguments
and correct datatypes.
"""
function argtable()
    settings = ArgParseSettings()
    defaults = load_defaults()

    @add_arg_table settings begin
        # meta
        "--experiment"
        help = "The overall experiment name."
        arg_type = String
        default = defaults["experiment"] |> String

        "--name"
        help = "Name of a single experiment instance."
        arg_type = String
        default = defaults["name"] |> String

        "--run", "-r"
        help = "The run ID."
        arg_type = Int
        default = defaults["run"] |> Int

        "--scalar_saving_interval"
        help = "The interval at which scalar quantities are stored measured in epochs."
        arg_type = Int
        default = defaults["scalar_saving_interval"] |> Int

        "--image_saving_interval"
        help = "The interval at which images are stored measured in epochs."
        arg_type = Int
        default = defaults["image_saving_interval"] |> Int

        # data
        "--path_to_data", "-d"
        help = "Path to (folder of) dataset used for training."
        arg_type = String
        default = defaults["path_to_data"] |> String

        "--path_to_inputs"
        help = "Path to (folder of) external inputs used for training."
        arg_type = String
        default = defaults["path_to_inputs"] |> String

        "--path_to_artifacts"
        help = "Path to (folder of) noise artifacts used for training."
        arg_type = String
        default = defaults["path_to_artifacts"] |> String

        "--data_id"
        help = "ID identifying the data and its inputs/artifacts.
        Can be left empty if one writes the full paths and not folder beforehand. "
        arg_type = String
        default = defaults["data_id"] |> String

        "--train_test_split"
        help = "Marks the endpoint of the trainset of the dataset as Integer or as ratio of the total dataset length as Float.
        If the integer is equal to the length of the total dataset there is no testset. Same applies for choosing 1."
        arg_type = Float32
        default = defaults["train_test_split"] |> Float32

        # training
        "--teacher_forcing_interval"
        help = "The teacher forcing interval to use."
        arg_type = Int
        default = defaults["teacher_forcing_interval"] |> Int

        "--weak_tf_alpha"
        help = "α used for weak TF."
        arg_type = Float32
        default = defaults["weak_tf_alpha"] |> Float32

        "--gaussian_noise_level"
        help = "Noise level of gaussian noise added to teacher signals."
        arg_type = Float32
        default = defaults["gaussian_noise_level"] |> Float32

        "--sequence_length", "-T"
        help = "Length of sequences sampled from the dataset during training."
        arg_type = Int
        default = defaults["sequence_length"] |> Int

        "--batch_size", "-S"
        help = "The number of sequences to pack into one batch."
        arg_type = Int
        default = defaults["batch_size"] |> Int

        "--epochs", "-e"
        help = "The number of epochs to train for."
        arg_type = Int
        default = defaults["epochs"] |> Int

        "--batches_per_epoch"
        help = "The number of batches processed in each epoch."
        arg_type = Int
        default = defaults["batches_per_epoch"] |> Int

        "--gradient_clipping_norm"
        help = "The norm at which to clip gradients during training."
        arg_type = Float32
        default = defaults["gradient_clipping_norm"] |> Float32

        "--optimizer"
        help = "The optimizer to use for SGD optimization. Must be one provided by Flux.jl."
        arg_type = String
        default = defaults["optimizer"] |> String

        "--start_lr"
        help = "Learning rate passed to the optimizer at the beginning of training."
        arg_type = Float32
        default = defaults["start_lr"] |> Float32

        "--end_lr"
        help = "Target learning rate at the end of training due to exponential decay."
        arg_type = Float32
        default = defaults["end_lr"] |> Float32

        "--device"
        help = "Training device to use."
        arg_type = String
        default = defaults["device"] |> String

        # model
        "--model", "-m"
        help = "RNN to use."
        arg_type = String
        default = defaults["model"] |> String

        "--hidden_layers"
        help = "RNN MLP hidden layer dimensions."
        arg_type = String
        default = defaults["hidden_layers"] |> String

        "--latent_dim", "-M"
        help = "RNN latent dimension."
        arg_type = Int
        default = defaults["latent_dim"] |> Int

        "--num_bases", "-B"
        help = "Number of bases to use in dendritic PLRNN"
        arg_type = Int
        default = defaults["num_bases"] |> Int

        "--observation_model", "-o"
        help = "Observation model to use."
        arg_type = String
        default = defaults["observation_model"] |> String

        # Manifold Attractor Regularization
        "--MAR_ratio"
        help = "Ratio of regularized states."
        arg_type = Float32
        default = defaults["MAR_ratio"] |> Float32

        "--MAR_lambda"
        help = "Regularization factor λ."
        arg_type = Float32
        default = defaults["MAR_lambda"] |> Float32

        "--lat_model_regularization"
        help = "Regularization λ for latent model parameters."
        arg_type = Float32
        default = defaults["lat_model_regularization"] |> Float32

        "--obs_model_regularization"
        help = "Regularization λ for observation model parameters."
        arg_type = Float32
        default = defaults["obs_model_regularization"] |> Float32

        # Metrics
        "--D_stsp_scaling"
        help = "GMM scaling parameter."
        arg_type = Float32
        default = defaults["D_stsp_scaling"] |> Float32

        "--D_stsp_bins"
        help = "Number of bins for D_stsp binning method."
        arg_type = Int
        default = defaults["D_stsp_bins"] |> Int

        "--PSE_smoothing"
        help = "Gaussian kernel smoothing σ for power spectrum smoothing."
        arg_type = Float32
        default = defaults["PSE_smoothing"] |> Float32

        "--PE_n"
        help = "n-step ahead prediction error."
        arg_type = Int
        default = defaults["PE_n"] |> Int

        "--hidden_dim"
        help = "hidden dimension for shallow PLRNN"
        arg_type = Int
        default = defaults["hidden_dim"] |> Int

        "--optional_model_args"
        help = "Optional model arguments."
        arg_type = Vector{String}
        default = defaults["optional_model_args"] |> Vector{String}

        #fMRI specifics
        "--TR"
        help = "Set to 0 to ignore. The time resolution of the fMRI timeseries, i.e. the time between two images (in seconds)"
        arg_type = Float32
        default = defaults["TR"] |> Float32

        "--cut_l"
        help = "Set to 0 to ignore. How much in terms of length of the hrf is cropped from the deconvoluded data at the beginning.
         Choose Float between 0 and 1 to interpret it as ratio. If bigger than 1 it will be interpreted as integer"
        arg_type = Float32
        default = defaults["cut_l"] |> Float32

        "--cut_r"
        help = "Set to 0 to ignore. How much in terms of length of the hrf is cropped from the deconvoluded data at the end. 
        Choose Float between 0 and 1 to interpret it as ratio. If bigger than 1 it will be interpreted as integer"
        arg_type = Float32
        default = defaults["cut_r"] |> Float32

        "--min_conv_noise"
        help = "If the noise level estimated for the Wiener deconvolution is smaller than the min_conv_noise,
        the min_conv_noise is set to be the noise level. I.e. choosing 0 has no influence."
        arg_type = Float32
        default = defaults["min_conv_noise"] |> Float32

    end
    return settings
end

"""
    parse_commandline()

Parses all commandline arguments for execution of `main.jl`.
"""
parse_commandline() = parse_args(argtable())
