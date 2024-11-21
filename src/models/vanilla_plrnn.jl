using Flux: @functor
using Statistics: mean

using ..Datasets
using ..ObservationModels: ObservationModel, init_state

# abstract type
abstract type AbstractPLRNN end
(m::AbstractPLRNN)(z::AbstractVecOrMat) = step(m, z)
(m::AbstractPLRNN)(z::AbstractVecOrMat, s::AbstractVecOrMat) = step(m, z, s)
step(m::AbstractPLRNN, z::AbstractVecOrMat, s::AbstractVecOrMat) = step(m, z) + m.C * s
jacobian(m::AbstractPLRNN, z::AbstractVector) = Flux.jacobian(z -> m(z), z)[1]
jacobian(m::AbstractPLRNN, z::AbstractMatrix) = jacobian.([m], eachcol(z))

abstract type AbstractVanillaPLRNN <: AbstractPLRNN end

mutable struct PLRNN{V <: AbstractVector, M <: AbstractMatrix} <: AbstractVanillaPLRNN
    A::V
    W::M
    h::V
    C::Maybe{M}
end
@functor PLRNN

# initialization/constructor
PLRNN(M::Int) = PLRNN(initialize_A_W_h(M)..., nothing)
PLRNN(M::Int, K::Int) = PLRNN(initialize_A_W_h(M)..., Flux.glorot_uniform(M, K))

"""
    step(model, z)

Evolve `z` in time for one step according to the model `m` (equation).

`z` is either a `M` dimensional column vector or a `M x S` matrix, where `S` is
the batch dimension.

"""
step(m::PLRNN, z::AbstractVecOrMat) = m.A .* z .+ m.W * relu.(z) .+ m.h
jacobian(m::PLRNN, z::AbstractVector) = Diagonal(m.A) + m.W * Diagonal(z .> 0)

# mean-centered PLRNN
mutable struct mcPLRNN{V <: AbstractVector, M <: AbstractMatrix} <: AbstractVanillaPLRNN
    A::V
    W::M
    h::V
    C::Maybe{M}
end
@functor mcPLRNN

# initialization/constructor
mcPLRNN(M::Int) = mcPLRNN(initialize_A_W_h(M)..., nothing)
mcPLRNN(M::Int, K::Int) = mcPLRNN(initialize_A_W_h(M)..., Flux.glorot_uniform(M, K))

mean_center(z::AbstractVecOrMat) = z .- mean(z, dims = 1)
step(m::mcPLRNN, z::AbstractVecOrMat) = m.A .* z .+ m.W * relu.(mean_center(z)) .+ m.h
function jacobian(m::mcPLRNN, z::AbstractVector)
    M, type = length(z), eltype(z)
    ℳ = type(1 / M) * (M * I - ones(type, M, M))
    return Diagonal(m.A) + m.W * Diagonal(ℳ * z .> 0) * ℳ
end

### TODO: Move generate methods to a separate file
### TODO: Adopt N x T data convention (should actually be faster in Julia -> column major memory layout)

@inbounds """
    generate(model, z₁, T)

Generate a trajectory of length `T` using PLRNN model `m` given initial condition `z₁`.
"""
function generate(m::AbstractPLRNN, z₁::AbstractVector, T::Int)
    # trajectory placeholder
    Z = similar(z₁, T, length(z₁))
    # initial condition for model
    @views Z[1, :] .= z₁
    # evolve initial condition in time
    @views for t = 2:T
        Z[t, :] .= m(Z[t-1, :])
    end
    return Z
end

@inbounds function generate(m::AbstractPLRNN, z₁::AbstractVector, S::AbstractMatrix, T::Int)
    # trajectory placeholder
    Z = similar(z₁, T, length(z₁))
    # initial condition for model
    @views Z[1, :] .= z₁
    # evolve initial condition in time
    @views for t = 2:T
        Z[t, :] .= m(Z[t-1, :], S[t, :])
    end
    return Z
end

"""
    generate(model, observation_model, x₁,[S,] T)

Generate a trajectory of length `T` using PLRNN model `m` given initial condition `x₁`.
"""
function generate(d::Dataset, m::AbstractPLRNN, obs::ObservationModel, x₁::AbstractVector, T::Int)
    z₁ = init_state(obs, x₁)
    ts = generate(m, z₁, T)
    return permutedims(obs(ts'), (2, 1))
end

function generate(
    d::ExternalInputsDataset,
    m::AbstractPLRNN,
    obs::ObservationModel,
    x₁::AbstractVector,
    S::AbstractMatrix,
    T::Int,
)
    z₁ = init_state(obs, x₁)
    ts = generate(m, z₁, S, T)
    return permutedims(obs(ts'), (2, 1))
end

function generate(
    d::NuisanceArtifactsDataset,
    m::AbstractPLRNN,
    obs::ObservationModel,
    x₁::AbstractVector,
    R::AbstractMatrix,
    T::Int,
)
    z₁ = init_state(obs, x₁, R[1,:])
    ts = generate(m, z₁, T)
    return permutedims(obs(ts', R'), (2, 1))
end

function generate(
    d::ExternalInputsNuisanceArtifactsDataset,
    m::AbstractPLRNN,
    obs::ObservationModel,
    x₁::AbstractVector,
    S::AbstractMatrix,
    R::AbstractMatrix,
    T::Int,
)
    z₁ = init_state(obs, x₁, R[1,:])
    ts = generate(m, z₁, S, T)
    return permutedims(obs(ts', R'), (2, 1))
end

"""
    generate(model, observation_model, x₁,[S,][R,] T)
    for the convolutional datasets
"""
function generate(d::DatasetConv, m::AbstractPLRNN, obs::ObservationModel, x₁::AbstractVector, T::Int)
    z₁ = init_state(obs, x₁)
    ts = generate(m, z₁, T)
    ts_conv = hrf_conv(ts, d.hrf)
    return permutedims(obs(ts_conv'), (2, 1))
end

function generate(
    d::ExternalInputsDatasetConv,
    m::AbstractPLRNN,
    obs::ObservationModel,
    x₁::AbstractVector,
    S::AbstractMatrix,
    T::Int,
)
    z₁ = init_state(obs, x₁)
    ts = generate(m, z₁, S, T)
    ts_conv = hrf_conv(ts, d.hrf)
    return permutedims(obs(ts_conv'), (2, 1))
end

function generate(
    d::NuisanceArtifactsDatasetConv,
    m::AbstractPLRNN,
    obs::ObservationModel,
    x₁::AbstractVector,
    r₁::AbstractVector,
    R::AbstractMatrix,
    T::Int,
)
    z₁ = init_state(obs, x₁, r₁)
    ts = generate(m, z₁, T)
    ts_conv = hrf_conv(ts, d.hrf)
    return permutedims(obs(ts_conv', R'), (2, 1))
end

function generate(
    d::ExternalInputsNuisanceArtifactsDatasetConv,
    m::AbstractPLRNN,
    obs::ObservationModel,
    x₁::AbstractVector,
    r₁::AbstractVector,
    S::AbstractMatrix,
    R::AbstractMatrix,
    T::Int,
)
    z₁ = init_state(obs, x₁, r₁)
    ts = generate(m, z₁, S, T)
    ts_conv = hrf_conv(ts, d.hrf)
    return permutedims(obs(ts_conv', R'), (2, 1))
end
