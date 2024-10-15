using Flux
using ..PLRNNs
using ..ObservationModels
using ..Datasets

@inbounds function prediction_error(d::Dataset, model, O::ObservationModel, X::AbstractMatrix, n::Int)
    T = size(X, 1)
    T̃ = T - n

    # batched computations from here on
    # time dim -> batch dim
    z = @views init_state(O, X[1:T̃, :]')
    for _ = 1:n
        z = model(z)
    end

    # compute MSE
    mse = @views Flux.mse(O(z)', X[n+1:end, :])
    return mse
end

@inbounds function prediction_error(
    d::ExternalInputsDataset,
    model,
    O::ObservationModel,
    X::AbstractMatrix,
    S::AbstractMatrix,
    n::Int,
)
    T = size(X, 1)
    T̃ = T - n

    # batched computations from here on
    # time dim -> batch dim
    z = @views init_state(O, X[1:T̃, :]')
    for i = 1:n
        z = @views model(z, S[i+1:T̃+i, :]')
    end

    # compute MSE
    mse = @views Flux.mse(O(z)', X[n+1:end, :])
    return mse
end


@inbounds function prediction_error(
    d::NuisanceArtifactsDataset,
    model,
    O::ObservationModel,
    X::AbstractMatrix,
    R::AbstractMatrix,
    n::Int,
)
    T = size(X, 1)
    T̃ = T - n

    # batched computations from here on
    # time dim -> batch dim
    z = @views init_state(O, X[1:T̃, :]', R[1:T̃, :]')
    for i = 1:n
        z = @views model(z)
    end

    # compute MSE
    mse = @views Flux.mse(O(z, R[n+1:end, :]')', X[n+1:end, :])
    return mse
end

@inbounds function prediction_error(
    d::ExternalInputsNuisanceArtifactsDataset,
    model,
    O::ObservationModel,
    X::AbstractMatrix,
    S::AbstractMatrix,
    R::AbstractMatrix,
    n::Int,
)
    T = size(X, 1)
    T̃ = T - n

    # batched computations from here on
    # time dim -> batch dim
    z = @views init_state(O, X[1:T̃, :]', R[1:T̃, :]')
    for i = 1:n
        z = @views model(z, S[i+1:T̃+i, :]')
    end

    # compute MSE
    mse = @views Flux.mse(O(z, R[n+1:end, :]')', X[n+1:end, :])
    return mse
end
