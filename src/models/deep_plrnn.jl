using Flux: @functor

# abstract type
abstract type AbstractDeepPLRNN <: AbstractPLRNN end

# deep PLRNN 
# inspired by R Pascanu, C Gulcehre, K Cho, Y Bengio
# arXiv preprint arXiv:1312.6026, 2013 - arxiv.org
# "How to construct deep recurrent neural networks"

function string_to_vec(string::String)
    layers = []
    for layer in split(string, ",")
        push!(layers, parse(Int64, String(layer)))
    end
    return layers
end

function initLayers(M::Int, layers::AbstractVector)
    MLPlayers = []
    nₗ = length(layers)
    if nₗ == 0
        push!(MLPlayers, Identity)
    elseif nₗ == 1
        push!(MLPlayers, Dense(M, layers[1], relu))
        push!(MLPlayers, Dense(layers[1], M))
    elseif nₗ > 1
        push!(MLPlayers, Dense(M, layers[1], relu))
        for i = 2:nₗ
            push!(MLPlayers, Dense(layers[i-1], layers[i], relu))
        end
        push!(MLPlayers, Dense(layers[nₗ], M))
    end
    return Chain(MLPlayers)
end

mutable struct deepPLRNN{V <: AbstractVector, M <: AbstractMatrix} <: AbstractDeepPLRNN
    A::V
    MLP::Chain
    C::Maybe{M}
end
@functor deepPLRNN

# initialization/constructor
function deepPLRNN(M::Int64, layers::String)
    layers = string_to_vec(layers)
    MLP = initLayers(M, layers)
    A, _, _ = initialize_A_W_h(M)
    return deepPLRNN(A, MLP, nothing)
end

function deepPLRNN(M::Int64, layers::String, K::Int64)
    layers = string_to_vec(layers)
    MLP = initLayers(M, layers)
    A, _, _ = initialize_A_W_h(M)
    C = Flux.glorot_uniform(M, K)
    return deepPLRNN(A, MLP, C)
end

# initialization/constructor
#function deepPLRNN{T}(M::Int, layers::AbstractVector{Int}, N::Int) where T
#   AW = normalized_positive_definite(M; eltype=T)
#  A, W = diag(AW), offdiagonal(AW)
# MLP = initLayers(M, layers)
#h = zeros(T, M)
# L = uniform_init((M-N, N); eltype=T)

#return deepPLRNN(A, W, MLP, h, L)
#end

"""
    step(model, z)

Evolve `z` in time for one step according to the model `m` (equation).

`z` is either a `M` dimensional column vector or a `M x S` matrix, where `S` is
the batch dimension.

"""
step(m::deepPLRNN, z::AbstractVecOrMat) = m.A .* z .+ m.MLP(z)