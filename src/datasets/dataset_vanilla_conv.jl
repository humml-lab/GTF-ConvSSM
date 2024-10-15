"""
    DatasetConv(args; kwargs)

Standard dataset storing a continuous time series + deconvoluted version of
size T × N, where N is the data dimension.
"""
struct DatasetConv{M <: AbstractMatrix, v <: AbstractVector} <: AbstractDataset
    X::M
    X_test::M
    X_deconv::M
    X_deconv_test::M
    hrf::v
    name::String
end

function DatasetConv(path::String, data_id::String, train_test_split::Real, TR::Float32, cut_l::Float32,  cut_r::Float32, min_conv_noise::Float32, name::String; device = cpu, dtype = Float32)
    X_full = npzread(path_with_id(path,data_id)) .|> dtype |> device
    train_test_split = (train_test_split<=1 ? Int(floor(train_test_split*size(X_full,1))) : Int(train_test_split))
    
    #split total dataset in train and test set
    X = X_full[1:train_test_split, :] #use as dataset only trainset
    X_test = X_full[train_test_split+1:end, :]

    #hemodynamic response function (HRF)
    hrf = spm_hrf(TR) .|> dtype |> device
    X_deconv, X_deconv_test = get_deconv_data(path, data_id, train_test_split, cut_l, cut_r, min_conv_noise, hrf, X_full) |> device
   
    @assert ndims(X_full) == 2 "Data must be 2-dimensional but is $(ndims(X))-dimensional."
    return DatasetConv(X, X_test, X_deconv, X_deconv_test, hrf, name)
end

DatasetConv(path::String, data_id::String, train_test_split::Real, TR::Float32, cut_l::Float32, cut_r::Float32, min_conv_noise; device = cpu, dtype = Float32) =
    DatasetConv(path, data_id, train_test_split, TR, cut_l, cut_r, min_conv_noise,""; device = device, dtype = dtype)


@inbounds """
    sample_sequence(dataset, sequence_length)

Sample a sequence of length `T̃` from a time series X.
"""
function sample_sequence(D::DatasetConv, T̃::Int)
    T = size(D.X, 1)
    cut_l = findfirst(!isnan,D.X_deconv[:,1])-1
    i = rand(1+cut_l:T-T̃-1)
    return D.X[i:i+T̃, :], D.X_deconv[i:i+T̃, :]
end

@inbounds """
    sample_batch(dataset, seq_len, batch_size)

Sample a batch of sequences of batch size `S` from time series X
(with replacement!).
"""
function sample_batch(D::DatasetConv, T̃::Int, S::Int)
    N = size(D.X, 2)
    Xs = similar(D.X, N, S, T̃ + 1)
    Xs_deconv = similar(D.X, N, S, T̃ + 1)
    Threads.@threads for i = 1:S
        X̃, X̃_deconv = sample_sequence(D, T̃)
        Xs[:, i, :] .= X̃'
        Xs_deconv[:, i, :] .= X̃_deconv'
    end
    return Xs, Xs_deconv
end
