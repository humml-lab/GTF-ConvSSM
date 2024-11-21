abstract type RegressorObservationModel <: ObservationModel end

forward(O::RegressorObservationModel, z::AbstractVecOrMat, r::AbstractVecOrMat) =
    O.B * z + O.J * r
function forward(O::RegressorObservationModel, z::AbstractArray{T, 3},  r::AbstractArray{T, 3}) where {T}
    M, S, T̃ = size(z)
    P, S, T̃ = size(r)
    return reshape(O(reshape(z, M, :), reshape(r, P, :)), :, S, T̃)
end

inverse(O::RegressorObservationModel) = pinv(O.B)

apply_inverse(O::RegressorObservationModel, x::AbstractVecOrMat, r::AbstractVecOrMat) = 
(inverse(O) * (x .- O.J * r))
   
function apply_inverse(O::RegressorObservationModel, X::AbstractArray{T, 3}, R::AbstractArray) where {T} 
    N, S, T̃ = size(X)
    P, S, T̃ = size(R)
    return reshape(apply_inverse(O, reshape(X, N, :), reshape(R, P, :)), :, S, T̃)
end


"""
    Regressor(z)

Map latent states into observation space via an mapping x̂ₜ = Bzₜ + Jrₜ.

Accepted inputs for `z` are
- a `M`-dimensional `Vector` (a single latent state vector)
- a `M × S`-dimensional `Matrix` (a batch of latent state vectors)
- a `M × S × T̃`-dimensional `Array` (a sequence of batched state vectors) 
"""
mutable struct Regressor{M <: AbstractMatrix} <: RegressorObservationModel
    B::M
    J::M
end
Flux.trainable(O::Regressor) = (O.B, O.J)
Flux.@functor Regressor

Regressor(N::Int, M::Int, P::Int) =
    Regressor(Flux.glorot_uniform(N, M), Flux.glorot_uniform(N, P))

init_state(O::Regressor, x::AbstractVecOrMat, r::AbstractVecOrMat) = apply_inverse(O, x, r)
