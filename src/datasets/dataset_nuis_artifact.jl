"""
    NuisanceArtifactsDataset(args; kwargs)

Standard dataset storing a continuous time series of size
T × N, where N is the data dimension, and a corresponding time
series of nuisance artifacts of shape T × P.
"""
struct NuisanceArtifactsDataset{M <: AbstractMatrix} <: AbstractDataset
    X::M
    R::M
    X_test::M
    R_test::M
    name::String
end

function NuisanceArtifactsDataset(
    data_path::String,
    artifacts_path::String,
    data_id::String,
    train_test_split::Real,
    name::String;
    device = cpu,
    dtype = Float32,
)
    X_full = npzread(path_with_id(data_path,data_id)) .|> dtype |> device
    R_full = npzread(path_with_id(artifacts_path,data_id)) .|> dtype |> device

    train_test_split = (train_test_split<=1 ? Int(floor(train_test_split*size(X_full,1))) : Int(train_test_split))
    
    #split total dataset in train and test set
    X = X_full[1:train_test_split, :] #use as dataset only trainset
    R = R_full[1:train_test_split, :] #use as dataset only trainset
    X_test = X_full[train_test_split+1:end, :]
    R_test = R_full[train_test_split+1:end, :]


    @assert ndims(X_full) == ndims(R_full) == 2 "Data and artifacts must be 2D but are $(ndims(X))D and $(ndims(R))D."
    @assert size(X_full, 1) == size(R_full, 1) "Data and nuisance artifacts have to be of equal length."
    return NuisanceArtifactsDataset(X, R, X_test, R_test, name)
end

NuisanceArtifactsDataset(
    data_path::String,
    artifacts_path::String,
    data_id::String,
    train_test_split::Real;
    device = cpu,
    dtype = Float32,
) = NuisanceArtifactsDataset(data_path, artifacts_path, data_id, train_test_split,""; device = device, dtype = dtype)

@inbounds """
    sample_sequence(dataset, sequence_length)

Sample a sequence of length `T̃` from a time series X.
"""

@inbounds function sample_sequence(D::NuisanceArtifactsDataset, T̃::Int)
    T = size(D.X, 1)
    i = rand(1:T-T̃-1)
    return D.X[i:i+T̃, :], D.R[i:i+T̃, :]
end

@inbounds """
    sample_batch(dataset, seq_len, batch_size)

Sample a batch of sequences of batch size `S` from time series X
(with replacement!).
"""

@inbounds function sample_batch(D::NuisanceArtifactsDataset, T̃::Int, S::Int)
    N, P = size(D.X, 2), size(D.R, 2)
    Xs = similar(D.X, N, S, T̃ + 1)
    Rs = similar(D.X, P, S, T̃ + 1)
    Threads.@threads for i = 1:S
        X̃, R̃ = sample_sequence(D, T̃)
        Xs[:, i, :] .= X̃'
        Rs[:, i, :] .= R̃'
    end
    return Xs, Rs
end