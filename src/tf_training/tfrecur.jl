using Flux

using ..PLRNNs
using ..Utilities
using ..ObservationModels

abstract type AbstractTFRecur end
Flux.trainable(tfrec::AbstractTFRecur) = (tfrec.model, tfrec.O)

(tfrec::AbstractTFRecur)(d::Union{Dataset, DatasetConv}, X::AbstractArray{T, 3}) where {T} = forward(tfrec, d, X)
(tfrec::AbstractTFRecur)(d::Union{ExternalInputsDataset, ExternalInputsDatasetConv}, X::AbstractArray{T, 3}, S::AbstractArray{T, 3}) where {T} =
    forward(tfrec, d, X, S)
(tfrec::AbstractTFRecur)(d::Union{NuisanceArtifactsDataset, NuisanceArtifactsDatasetConv}, X::AbstractArray{T, 3}, R::AbstractArray{T, 3}) where {T} =
    forward(tfrec, d, X, R)
(tfrec::AbstractTFRecur)(d::Union{ExternalInputsNuisanceArtifactsDataset, ExternalInputsNuisanceArtifactsDatasetConv}, X::AbstractArray{T, 3}, S::AbstractArray{T, 3}, R::AbstractArray{T, 3}) where {T} =
    forward(tfrec, d, X, S, R)

"""
    forward(tfrec, X)

Forward pass using teacher forcing. If the latent dimension of
the RNN is larger than the dimension the observations live in, 
partial teacher forcing of the first `N = size(X, 1)` neurons is
used. Initializing latent state `z₁` is taken care of by the observation model.
"""
function forward(tfrec::AbstractTFRecur, d::Union{Dataset, DatasetConv}, X::AbstractArray{T, 3}) where {T}
    N, _, T̃ = size(X)
    M = size(tfrec.z, 1)

    # number of forced states
    D = min(N, M)

    # precompute forcing signals
    Z⃰ = apply_inverse(tfrec.O, X)

    # initialize latent state
    tfrec.z = @views init_state(tfrec.O, X[:, :, 1])

    # process sequence X
    Z = @views [tfrec(Z⃰[1:D, :, t], t) for t = 2:T̃]

    # reshape to 3d array and return
    return reshape(reduce(hcat, Z), size(tfrec.z)..., :)
end

"""
    forward(tfrec, X, S)

Forward pass using teacher forcing with external inputs.
"""
function forward(
    tfrec::AbstractTFRecur,
    d::Union{ExternalInputsDataset, ExternalInputsDatasetConv},
    X::AbstractArray{T, 3},
    S::AbstractArray{T, 3},
) where {T}
    N, _, T̃ = size(X)
    M = size(tfrec.z, 1)

    # number of forced states
    D = min(N, M)

    # precompute forcing signals
    Z⃰ = apply_inverse(tfrec.O, X)

    # initialize latent state
    tfrec.z = @views init_state(tfrec.O, X[:, :, 1])

    # process sequence X
    Z = @views [tfrec(Z⃰[1:D, :, t], S[:, :, t], t) for t = 2:T̃]

    # reshape to 3d array and return
    return reshape(reduce(hcat, Z), size(tfrec.z)..., :)
end

"""
    forward(tfrec, X, R)

Forward pass using teacher forcing with nuisance artifacts.
"""
function forward(
    tfrec::AbstractTFRecur,
    d::Union{NuisanceArtifactsDataset, NuisanceArtifactsDatasetConv},
    X::AbstractArray{T, 3},
    R::AbstractArray{T, 3},
) where {T}
    N, _, T̃ = size(X)
    M = size(tfrec.z, 1)

    # number of forced states
    D = min(N, M)

    # precompute forcing signals
    Z⃰ = apply_inverse(tfrec.O, X, R)

    # initialize latent state
    tfrec.z = @views init_state(tfrec.O, X[:, :, 1], R[:, :, 1])

    # process sequence X
    Z = @views [tfrec(Z⃰[1:D, :, t], t) for t = 2:T̃]

    # reshape to 3d array and return
    return reshape(reduce(hcat, Z), size(tfrec.z)..., :)
end

"""
    forward(tfrec, X, S, R)

Forward pass using teacher forcing with external inputs and nuisance artifacts.
"""
function forward(
    tfrec::AbstractTFRecur,
    d::Union{ExternalInputsNuisanceArtifactsDataset, ExternalInputsNuisanceArtifactsDatasetConv},
    X::AbstractArray{T, 3},
    S::AbstractArray{T, 3},
    R::AbstractArray{T, 3},
) where {T}
    N, _, T̃ = size(X)
    M = size(tfrec.z, 1)

    # number of forced states
    D = min(N, M)

    # precompute forcing signals
    Z⃰ = apply_inverse(tfrec.O, X, R)

    # initialize latent state
    tfrec.z = @views init_state(tfrec.O, X[:, :, 1], R[:, :, 1])

    # process sequence X
    Z = @views [tfrec(Z⃰[1:D, :, t], S[:, :, t], t) for t = 2:T̃]

    # reshape to 3d array and return
    return reshape(reduce(hcat, Z), size(tfrec.z)..., :)
end

"""
Inspired by `Flux.Recur` struct, which by default has no way
of incorporating teacher forcing.

This is just a convenience wrapper around stateful models,
to be used during training.
"""
mutable struct TFRecur{M <: AbstractMatrix, 𝒪 <: ObservationModel} <: AbstractTFRecur
    # stateful model, e.g. PLRNN
    model::Any
    # observation model
    O::𝒪
    # state of the model
    z::M
    # forcing interval
    const τ::Int
end
Flux.@functor TFRecur

function (tfrec::TFRecur)(x::AbstractMatrix, t::Int)
    # determine if it is time to force the model
    z = tfrec.z

    # perform one step using the model, update model state
    z = tfrec.model(z)

    # force
    z̃ = (t - 1) % tfrec.τ == 0 ? force(z, x) : z
    tfrec.z = z̃
    return z
end

function (tfrec::TFRecur)(x::AbstractMatrix, s::AbstractMatrix, t::Int)
    # determine if it is time to force the model
    z = tfrec.z

    # perform one step using the model, update model state
    z = tfrec.model(z, s)

    # force
    z̃ = (t - 1) % tfrec.τ == 0 ? force(z, x) : z
    tfrec.z = z̃
    return z
end

# Weak TF Recur
mutable struct WeakTFRecur{M <: AbstractMatrix, ℳ <: AbstractPLRNN, 𝒪 <: ObservationModel} <: AbstractTFRecur
    # stateful model, e.g. PLRNN
    model::ℳ
    # ObservationModel
    O::𝒪
    # state of the model
    z::M
    # weak forcing α
    const α::Float32
end
Flux.@functor WeakTFRecur

function (tfrec::WeakTFRecur)(z⃰::AbstractMatrix, t::Int)
    z = tfrec.z
    D, M = size(z⃰, 1), size(z, 1)
    z = tfrec.model(z)
    # weak tf
    z̃ = @views force(z[1:D, :], z⃰, tfrec.α)
    z̃ = (D == M) ? z̃ : force(z, z̃)

    tfrec.z = z̃
    return z
end

function (tfrec::WeakTFRecur)(z⃰::AbstractMatrix, s::AbstractMatrix, t::Int)
    z = tfrec.z
    D, M = size(z⃰, 1), size(z, 1)
    z = tfrec.model(z, s)
    # weak tf
    z̃ = @views force(z[1:D, :], z⃰, tfrec.α)
    z̃ = (D == M) ? z̃ : force(z, z̃)

    tfrec.z = z̃
    return z
end
