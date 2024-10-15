"""
    ExternalInputsDataset(args; kwargs)

Standard dataset storing a continuous time series of size
T × N, where N is the data dimension, and a corresponding time
series of exogeneous inputs of shape T × K.
"""
struct ExternalInputsDataset{M <: AbstractMatrix} <: AbstractDataset
    X::M
    S::M
    X_test::M
    S_test::M
    name::String
end

function ExternalInputsDataset(
    data_path::String,
    inputs_path::String,
    data_id::String,
    train_test_split::Real,
    name::String;
    device = cpu,
    dtype = Float32,
)
    X_full = npzread(path_with_id(data_path,data_id)) .|> dtype |> device
    S_full = npzread(path_with_id(inputs_path,data_id)) .|> dtype |> device

    train_test_split = (train_test_split<=1 ? Int(floor(train_test_split*size(X_full,1))) : Int(train_test_split))

    #split total dataset in train and test set
    X = X_full[1:train_test_split, :] #use as dataset only trainset
    S = S_full[1:train_test_split, :] #use as dataset only trainset
    X_test = X_full[train_test_split+1:end, :]
    S_test = S_full[train_test_split+1:end, :]

    @assert ndims(X_full) == ndims(S_full) == 2 "Data and inputs must be 2D but are $(ndims(X))D and $(ndims(S))D."
    @assert size(X_full, 1) == size(S_full, 1) "Data and exogeneous inputs have to be of equal length."
    return ExternalInputsDataset(X, S, X_test, S_test, name)
end

ExternalInputsDataset(
    data_path::String,
    inputs_path::String,
    data_id::String,
    train_test_split::Real;
    device = cpu,
    dtype = Float32,
) = ExternalInputsDataset(data_path, inputs_path, data_id, train_test_split,""; device = device, dtype = dtype)

@inbounds """
    sample_sequence(dataset, sequence_length)

Sample a sequence of length `T̃` from a time series X.
"""

@inbounds function sample_sequence(D::ExternalInputsDataset, T̃::Int)
    T = size(D.X, 1)
    i = rand(1:T-T̃-1)
    return D.X[i:i+T̃, :], D.S[i:i+T̃, :]
end

@inbounds """
    sample_batch(dataset, seq_len, batch_size)

Sample a batch of sequences of batch size `S` from time series X
(with replacement!).
"""

@inbounds function sample_batch(D::ExternalInputsDataset, T̃::Int, S::Int)
    N, K = size(D.X, 2), size(D.S, 2)
    Xs = similar(D.X, N, S, T̃ + 1)
    Ss = similar(D.X, K, S, T̃ + 1)
    Threads.@threads for i = 1:S
        X̃, S̃ = sample_sequence(D, T̃)
        Xs[:, i, :] .= X̃'
        Ss[:, i, :] .= S̃'
    end
    return Xs, Ss
end