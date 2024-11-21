abstract type AbstractDataset end
"""
    Dataset(args; kwargs)

Standard dataset storing a continuous time series of
size T × N, where N is the data dimension.
"""
struct Dataset{M <: AbstractMatrix} <: AbstractDataset
    X::M
    X_test::M
    name::String
end

function Dataset(path::String, data_id::String, train_test_split::Real, name::String; device = cpu, dtype = Float32)
    X_full = npzread(path_with_id(path,data_id)) .|> dtype |> device
    train_test_split = (train_test_split<=1 ? Int(floor(train_test_split*size(X_full,1))) : Int(train_test_split))
    
    #split total dataset in train and test set
    X = X_full[1:train_test_split, :] #use as dataset only trainset
    X_test = X_full[train_test_split+1:end, :]

    @assert ndims(X) == 2 "Data must be 2-dimensional but is $(ndims(X))-dimensional."
    return Dataset(X, X_test, name)
end

Dataset(path::String, data_id::String, train_test_split::Real; device = cpu, dtype = Float32) =
    Dataset(path, data_id, train_test_split,""; device = device, dtype = dtype)


@inbounds """
    sample_sequence(dataset, sequence_length)

Sample a sequence of length `T̃` from a time series X.
"""
function sample_sequence(D::Dataset, T̃::Int)
    T = size(D.X, 1)
    i = rand(1:T-T̃-1)
    return D.X[i:i+T̃, :]
end

@inbounds """
    sample_batch(dataset, seq_len, batch_size)

Sample a batch of sequences of batch size `S` from time series X
(with replacement!).
"""
function sample_batch(D::Dataset, T̃::Int, S::Int)
    N = size(D.X, 2)
    Xs = similar(D.X, N, S, T̃ + 1)
    Threads.@threads for i = 1:S
        Xs[:, i, :] .= sample_sequence(D, T̃)'
    end
    return Xs
end
