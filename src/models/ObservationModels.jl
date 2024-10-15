module ObservationModels

using ..Utilities

abstract type ObservationModel end
(O::ObservationModel)(z::AbstractArray) = forward(O, z)
(O::ObservationModel)(z::AbstractArray, r::AbstractArray) = forward(O, z, r)

export ObservationModel, Identity, Affine, Regressor, apply_inverse, init_state

include("affine.jl")
include("regressor.jl")

end