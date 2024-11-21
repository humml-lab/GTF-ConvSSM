using Flux
using ..PLRNNs
using ..ObservationModels
using ..Datasets

@inbounds function prediction_error(
    d::DatasetConv,
    model,
    O::ObservationModel,
    n::Int)

    cutl = findfirst(!isnan, d.X_deconv[:,1])-1
    cutr = findfirst(!isnan, d.X_deconv[end:-1:1,1])-1

    X_deconv_full = cat(d.X_deconv[1:end-cutr,:], d.X_deconv_test, d.X_deconv[end-cutr+1:end,:] , dims=1)

    T = size(d.X, 1) #corresponds to train_test_split as well
    #if cutr is bigger than n than one has to determine the last non-Nan value in X_deconv with cutr.
    T̃ = T - max(cutr,n)

    # batched computations from here on
    # time dim -> batch dim
    z = @views init_state(O, d.X_deconv[cutl+1:T̃, :]') #cutl +1 first non-Nan value
    #try to predict all X_test values(but if n<cut_r, not enough initial conditions at the end to access)
    #T+1 = start of test set, length(hrf)-1 states "get lost" during convolution and n time steps we evolve
    z_test = @views init_state(O, X_deconv_full[T+1-length(d.hrf)+1-n:end-max(cutr,n), :]')
    for _ = 1:n
        z = model(z)
        z_test = model(z_test)
    end
    #convolution
    z_conv = hrf_conv(z', d.hrf)
    z_conv_test = hrf_conv(z_test', d.hrf)
    # compute MSE
    mse = @views Flux.mse(O(z_conv')', d.X[cutl+1 + n + length(d.hrf)-1:end-relu(cutr-n), :])
    mse_test = @views Flux.mse(O(z_conv_test')',d.X_test[1:end-relu(cutr-n), :])
    return mse, mse_test
end

@inbounds function prediction_error(
    d::ExternalInputsDatasetConv,
    model,
    O::ObservationModel,
    n::Int,
)
    S_full = cat(d.S, d.S_test, dims=1)

    cutl = findfirst(!isnan, d.X_deconv[:,1])-1
    cutr = findfirst(!isnan, d.X_deconv[end:-1:1,1])-1

    X_deconv_full = cat(d.X_deconv[1:end-cutr,:], d.X_deconv_test, d.X_deconv[end-cutr+1:end,:] , dims=1)

    T = size(d.X, 1)
    #if cutr is bigger than n than one has to determine the last non-Nan value in X_deconv with cutr.
    T̃ = T - max(cutr,n)

    # batched computations from here on
    # time dim -> batch dim
    z = @views init_state(O, d.X_deconv[cutl+1:T̃, :]')
    z_test = @views init_state(O, X_deconv_full[T+1-length(d.hrf)+1-n:end-max(cutr,n), :]')
    for i = 1:n
        z = @views model(z, d.S[cutl+1+i:T̃+i, :]')
        z_test = @views model(z_test, S_full[T+1-length(d.hrf)+1-n+i:end-max(cutr,n)+i, :]')
    end
    #convolution
    z_conv = hrf_conv(z',d.hrf)
    z_conv_test = hrf_conv(z_test',d.hrf)
    # compute MSE
    mse = @views Flux.mse(O(z_conv')', d.X[cutl+1 + n + length(d.hrf)-1:end-relu(cutr-n), :])
    mse_test = @views Flux.mse(O(z_conv_test')', d.X_test[1:end-relu(cutr-n), :])
    return mse, mse_test
end


@inbounds function prediction_error(
    d::NuisanceArtifactsDatasetConv,
    model,
    O::ObservationModel,
    n::Int,
)
    cutl = findfirst(!isnan, d.X_deconv[:,1])-1
    cutr = findfirst(!isnan, d.X_deconv[end:-1:1,1])-1

    X_deconv_full = cat(d.X_deconv[1:end-cutr,:], d.X_deconv_test, d.X_deconv[end-cutr+1:end,:] , dims=1)
    R_deconv_full = cat(d.R_deconv[1:end-cutr,:], d.R_deconv_test, d.R_deconv[end-cutr+1:end,:] , dims=1)

    T = size(d.X, 1)
    T̃ = T - max(cutr,n)

    # batched computations from here on
    # time dim -> batch dim
    z = @views init_state(O, d.X_deconv[cutl+1:T̃, :]', d.R_deconv[cutl+1:T̃, :]')
    z_test = @views init_state(O, X_deconv_full[T+1-length(d.hrf)+1-n:end-max(cutr,n), :]', R_deconv_full[T+1-length(d.hrf)+1-n:end-max(cutr,n), :]')
    for _ = 1:n
        z = @views model(z)
        z_test = @views model(z_test)
    end
    #convolution
    z_conv = hrf_conv(z',d.hrf)
    z_conv_test = hrf_conv(z_test', d.hrf)
    # compute MSE
    mse = @views Flux.mse(O(z_conv', d.R[cutl+1 + n + length(d.hrf)-1:end-relu(cutr-n), :]')', d.X[cutl+1 + n + length(d.hrf)-1:end-relu(cutr-n), :])
    mse_test = @views Flux.mse(O(z_conv_test', d.R_test[1:end-relu(cutr-n), :]')',d.X_test[1:end-relu(cutr-n), :])
    return mse, mse_test
end

@inbounds function prediction_error(
    d::ExternalInputsNuisanceArtifactsDatasetConv,
    model,
    O::ObservationModel,
    n::Int,
)
    S_full = cat(d.S, d.S_test, dims=1)

    cutl = findfirst(!isnan, d.X_deconv[:,1])-1
    cutr = findfirst(!isnan, d.X_deconv[end:-1:1,1])-1

    X_deconv_full = cat(d.X_deconv[1:end-cutr,:], d.X_deconv_test, d.X_deconv[end-cutr+1:end,:] , dims=1)
    R_deconv_full = cat(d.R_deconv[1:end-cutr,:], d.R_deconv_test, d.R_deconv[end-cutr+1:end,:] , dims=1)
    
    T = size(d.X, 1)
    T̃ = T - max(cutr,n)

    # batched computations from here on
    # time dim -> batch dim
    z = @views init_state(O, d.X_deconv[cutl+1:T̃, :]', d.R_deconv[cutl+1:T̃, :]')
    z_test = @views init_state(O, X_deconv_full[T+1-length(d.hrf)+1-n:end-max(cutr,n), :]', R_deconv_full[T+1-length(d.hrf)+1-n:end-max(cutr,n), :]')
    for i = 1:n
        z = @views model(z, d.S[cutl+1+i:T̃+i, :]')
        z_test = @views model(z_test, S_full[T+1-length(d.hrf)+1-n+i:end-max(cutr,n)+i, :]')
    end
    #convolution
    z_conv = hrf_conv(z',d.hrf)
    z_conv_test = hrf_conv(z_test',d.hrf)
    # compute MSE
    mse = @views Flux.mse(O(z_conv', d.R[cutl+1 + n + length(d.hrf)-1:end-relu(cutr-n), :]')', d.X[cutl+1 + n + length(d.hrf)-1:end-relu(cutr-n), :])
    mse_test = @views Flux.mse(O(z_conv_test', d.R_test[1:end-relu(cutr-n), :]')',d.X_test[1:end-relu(cutr-n), :])
    return mse, mse_test
end