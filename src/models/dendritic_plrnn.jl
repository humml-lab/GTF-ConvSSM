using Flux: @functor
using Tullio
using CUDA

# abstract type
abstract type AbstractDendriticPLRNN <: AbstractPLRNN end
step(m::AbstractDendriticPLRNN, z::AbstractVecOrMat) = m.A .* z .+ m.W * Φ(m, z) .+ m.h

"""
    dendPLRNN

Implementation of the dendritic PLRNN introduced in 
https://proceedings.mlr.press/v162/brenner22a.html.
"""
mutable struct dendPLRNN{V <: AbstractVector, M <: AbstractMatrix} <: AbstractDendriticPLRNN
    A::V
    W::M
    h::V
    α::V
    H::M
    C::Maybe{M}
end
@functor dendPLRNN

# initialization/constructor
function dendPLRNN(M::Int, B::Int)
    A, W, h = initialize_A_W_h(M)
    α = uniform_init((B,))
    H = randn(Float32, M, B)
    return dendPLRNN(A, W, h, α, H, nothing)
end

function dendPLRNN(M::Int, B::Int, X::AbstractMatrix)
    A, W, h = initialize_A_W_h(M)
    α = uniform_init((B,))
    H = uniform_threshold_init((M, B), X)
    return dendPLRNN(A, W, h, α, H, nothing)
end

function dendPLRNN(M::Int, B::Int, K::Int)
    A, W, h = initialize_A_W_h(M)
    α = uniform_init((B,))
    H = randn(Float32, M, B)
    C = Flux.glorot_uniform(M, K)
    return dendPLRNN(A, W, h, α, H, C)
end

function dendPLRNN(M::Int, B::Int, X::AbstractMatrix, K::Int)
    A, W, h = initialize_A_W_h(M)
    α = uniform_init((B,))
    H = uniform_threshold_init((M, B), X)
    C = Flux.glorot_uniform(M, K)
    return dendPLRNN(A, W, h, α, H, C)
end

Φ(m::dendPLRNN, z::AbstractVecOrMat) = basis_expansion(z, m.α, m.H)

basis_expansion(z::AbstractMatrix, α::AbstractVector, H::AbstractMatrix) =
    @tullio z̃[m, s] := α[b] * relu(z[m, s] - H[m, b])

basis_expansion(z::AbstractVector, α::AbstractVector, H::AbstractMatrix) =
    @tullio z̃[m] := α[b] * relu(z[m] - H[m, b])

function jacobian(m::dendPLRNN, z::AbstractVector)
    @tullio ∂Φ_∂z[m, m] := m.α[b] * (z[m] > m.H[m, b])
    return Diagonal(m.A) + m.W * ∂Φ_∂z
end

################################################

"""
    clippedDendPLRNN

State clipping formulation the the `dendPLRNN`, which guarantees bounded orbits if
||A||₂ < 1, where ||⋅||₂ is the spectral norm.
"""
mutable struct clippedDendPLRNN{V <: AbstractVector, M <: AbstractMatrix} <:
               AbstractDendriticPLRNN
    A::V
    W::M
    h::V
    α::V
    H::M
    C::Maybe{M}
end
@functor clippedDendPLRNN

# initialization/constructor
function clippedDendPLRNN(M::Int, B::Int)
    A, W, h = initialize_A_W_h(M)
    α = uniform_init((B,))
    H = randn(Float32, M, B)
    return clippedDendPLRNN(A, W, h, α, H, nothing)
end

function clippedDendPLRNN(M::Int, B::Int, X::AbstractMatrix)
    A, W, h = initialize_A_W_h(M)
    α = uniform_init((B,))
    H = uniform_threshold_init((M, B), X)
    return clippedDendPLRNN(A, W, h, α, H, nothing)
end

function clippedDendPLRNN(M::Int, B::Int, K::Int)
    A, W, h = initialize_A_W_h(M)
    α = uniform_init((B,))
    H = randn(Float32, M, B)
    C = Flux.glorot_uniform(M, K)
    return clippedDendPLRNN(A, W, h, α, H, C)
end

function clippedDendPLRNN(M::Int, B::Int, X::AbstractMatrix, K::Int)
    A, W, h = initialize_A_W_h(M)
    α = uniform_init((B,))
    H = uniform_threshold_init((M, B), X)
    C = Flux.glorot_uniform(M, K)
    return clippedDendPLRNN(A, W, h, α, H, C)
end

Φ(m::clippedDendPLRNN, z::AbstractVecOrMat) = clipping_basis_expansion(z, m.α, m.H)

clipping_basis_expansion(z::AbstractMatrix, α::AbstractVector, H::AbstractMatrix) =
    @tullio z̃[m, s] := α[b] * (relu(z[m, s] - H[m, b]) - relu(z[m, s]))

clipping_basis_expansion(z::AbstractVector, α::AbstractVector, H::AbstractMatrix) =
    @tullio z̃[m] := α[b] * (relu(z[m] - H[m, b]) - relu(z[m]))

function jacobian(m::clippedDendPLRNN, z::AbstractVector)
    @tullio ∂Φ_∂z[m, m] := m.α[b] * ((z[m] > m.H[m, b]) - (z[m] > 0))
    return Diagonal(m.A) + m.W * ∂Φ_∂z
end

################################################

# fully connected dendritic PLRNN 
mutable struct FCDendPLRNN{
    V <: AbstractVector,
    M <: AbstractMatrix,
    Arr <: AbstractArray{Float32, 3},
} <: AbstractDendriticPLRNN
    A::V
    W::Arr
    h::V
    H::M
    C::Maybe{M}
end
@functor FCDendPLRNN

# initialization/constructor
function FCDendPLRNN(M::Int, B::Int)
    A, _, h = initialize_A_W_h(M)
    # initialize W
    W = Array{Float32, 3}(undef, B, M, M)
    for b = 1:B
        W[b, :, :] = offdiagonal(normalized_positive_definite(M))
    end
    H = randn(Float32, M, B)
    return FCDendPLRNN(A, W, h, H, nothing)
end

function FCDendPLRNN(M::Int, B::Int, X::AbstractMatrix)
    A, _, h = initialize_A_W_h(M)
    # initialize W
    W = Array{Float32, 3}(undef, B, M, M)
    for b = 1:B
        W[b, :, :] = offdiagonal(normalized_positive_definite(M))
    end
    H = uniform_threshold_init((M, B), X)
    return FCDendPLRNN(A, W, h, H, nothing)
end

function FCDendPLRNN(M::Int, B::Int, K::Int)
    A, _, h = initialize_A_W_h(M)
    # initialize W
    W = Array{Float32, 3}(undef, B, M, M)
    for b = 1:B
        W[b, :, :] = offdiagonal(normalized_positive_definite(M))
    end
    H = randn(Float32, M, B)
    C = Flux.glorot_uniform(M, K)
    return FCDendPLRNN(A, W, h, H, C)
end

function FCDendPLRNN(M::Int, B::Int, X::AbstractMatrix, K::Int)
    A, _, h = initialize_A_W_h(M)
    # initialize W
    W = Array{Float32, 3}(undef, B, M, M)
    for b = 1:B
        W[b, :, :] = offdiagonal(normalized_positive_definite(M))
    end
    H = uniform_threshold_init((M, B), X)
    C = Flux.glorot_uniform(M, K)
    return FCDendPLRNN(A, W, h, H, C)
end

step(m::FCDendPLRNN, z::AbstractVecOrMat) = m.A .* z .+ Φ(m, z) .+ m.h

Φ(m::FCDendPLRNN, z::AbstractVecOrMat) = fc_basis_expansion(z, m.W, m.H)

fc_basis_expansion(z::AbstractMatrix, W::AbstractArray{T, 3}, H::AbstractMatrix) where {T} =
    @tullio z̃[m, s] := W[b, m, k] * relu(z[k, s] - H[k, b])

fc_basis_expansion(z::AbstractVector, W::AbstractArray{T, 3}, H::AbstractMatrix) where {T} =
    @tullio z̃[m] := W[b, m, k] * relu(z[k] - H[k, b])

