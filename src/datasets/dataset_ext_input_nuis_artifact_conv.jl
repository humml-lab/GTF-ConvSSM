"""
    ExternalInputsNuisanceArtifactsDatasetConv(args; kwargs)

Standard dataset storing a continuous time series of size
T × N, where N is the data dimension, a corresponding time
series of exogeneous inputs of shape T × K  and a corresponding time
series of nuisance artifacts of shape T × P + needed deconvoluted series
"""
struct ExternalInputsNuisanceArtifactsDatasetConv{M <: AbstractMatrix, v <: AbstractVector} <: AbstractDataset
    X::M
    S::M
    R::M
    X_test::M
    S_test::M
    R_test::M
    X_deconv::M
    R_deconv::M
    X_deconv_test::M
    R_deconv_test::M
    hrf::v
    name::String
end

function ExternalInputsNuisanceArtifactsDatasetConv(
    data_path::String,
    inputs_path::String,
    artifacts_path::String,
    data_id::String,
    train_test_split::Real,
    TR::Float32,
    cut_l::Float32,
    cut_r::Float32,
    min_conv_noise::Float32,
    name::String;
    device = cpu,
    dtype = Float32,
)
    X_full = npzread(path_with_id(data_path,data_id)) .|> dtype |> device
    S_full = npzread(path_with_id(inputs_path,data_id)) .|> dtype |> device
    R_full = npzread(path_with_id(artifacts_path,data_id)) .|> dtype |> device

    train_test_split = (train_test_split<=1 ? Int(floor(train_test_split*size(X_full,1))) : Int(train_test_split))

    #split total dataset in train and test set
    X = X_full[1:train_test_split, :] #use as dataset only trainset
    S = S_full[1:train_test_split, :] #use as dataset only trainset
    R = R_full[1:train_test_split, :] #use as dataset only trainset
    X_test = X_full[train_test_split+1:end, :]
    S_test = S_full[train_test_split+1:end, :]
    R_test = R_full[train_test_split+1:end, :]

     #hemodynamic response function (HRF)
     hrf = spm_hrf(TR)
     X_deconv, X_deconv_test = get_deconv_data(data_path, data_id, train_test_split, cut_l, cut_r, min_conv_noise, hrf, X_full)
     R_deconv, R_deconv_test = get_deconv_data(artifacts_path, data_id, train_test_split, cut_l, cut_r, 0f0, hrf, R_full)

    @assert ndims(X_full) == ndims(S_full) == ndims(R_full) == 2 "Data, inputs and artifacts must be 2D but are $(ndims(X))D, $(ndims(S))D and $(ndims(R))D."
    @assert size(X_full, 1) == size(S_full, 1) == size(R_full, 1) "Data, exogeneous inputs and nuisance artifacts have to be of equal length."
    return ExternalInputsNuisanceArtifactsDatasetConv(X, S, R, X_test, S_test, R_test, X_deconv, R_deconv, X_deconv_test, R_deconv_test, hrf, name)
end

ExternalInputsNuisanceArtifactsDatasetConv(
    data_path::String,
    inputs_path::String,
    artifacts_path::String,
    data_id::String,
    train_test_split::Real,
    TR::Float32,
    cut_l::Float32,
    cut_r::Float32,
    min_conv_noise::Float32;
    device = cpu,
    dtype = Float32,
) = ExternalInputsNuisanceArtifactsDatasetConv(data_path, inputs_path, artifacts_path, data_id, train_test_split, TR, cut_l, cut_r, min_conv_noise,""; device = device, dtype = dtype)

@inbounds """
    sample_sequence(dataset, sequence_length)

Sample a sequence of length `T̃` from a time series X.
"""

@inbounds function sample_sequence(D::ExternalInputsNuisanceArtifactsDatasetConv, T̃::Int)
    T = size(D.X, 1)
    cut_l = findfirst(!isnan,D.X_deconv[:,1])-1
    i = rand(1+cut_l:T-T̃-1)
    return D.X[i:i+T̃, :], D.S[i:i+T̃, :], D.R[i:i+T̃, :], D.X_deconv[i:i+T̃, :], D.R_deconv[i:i+T̃, :]
end

@inbounds """
    sample_batch(dataset, seq_len, batch_size)

Sample a batch of sequences of batch size `S` from time series X
(with replacement!).
"""

@inbounds function sample_batch(D::ExternalInputsNuisanceArtifactsDatasetConv, T̃::Int, S::Int)
    N, K, P = size(D.X, 2), size(D.S, 2), size(D.R, 2)
    Xs = similar(D.X, N, S, T̃ + 1)
    Ss = similar(D.X, K, S, T̃ + 1)
    Rs = similar(D.X, P, S, T̃ + 1)
    Xs_deconv = similar(D.X, N, S, T̃ + 1)
    Rs_deconv = similar(D.X, P, S, T̃ + 1)
    Threads.@threads for i = 1:S
        X̃, S̃, R̃, X̃_deconv, R̃_deconv = sample_sequence(D, T̃)
        Xs[:, i, :] .= X̃'
        Ss[:, i, :] .= S̃'
        Rs[:, i, :] .= R̃'
        Xs_deconv[:, i, :] .= X̃_deconv'
        Rs_deconv[:, i, :] .= R̃_deconv'
    end
    return Xs, Ss, Rs, Xs_deconv, Rs_deconv
end