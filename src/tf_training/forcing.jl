using ChainRulesCore: NoTangent, @thunk
import ChainRulesCore

@inbounds """
    force(z, x)

Replace the first `N = dim(x)` dimensions of `z` with `x`. If `α` is provided,
weak teacher forcing is applied. Weak TF asserts that `N == M`.

Supplied with custom backward `ChainRulesCore.rrule`.
"""

function force(z::AbstractMatrix, x::AbstractMatrix)
    N = size(x, 1)
    #Do not force with NaN values in observation
    index = findall(!isnan, x)
    z[index] = x[index]
    return [z[1:N, :]; z[N+1:end, :]]
end

@inbounds function ChainRulesCore.rrule(
    ::typeof(force),
    z::AbstractMatrix,
    x::AbstractMatrix,
)
    N = size(x, 1)
    function force_pullback(ΔΩ)
        index = findall(!isnan, x)
        ∂z = ΔΩ
        ∂x = similar(ΔΩ[1:N, :])
        ∂x[index] = ΔΩ[index]
        # in-place here for speed
        ∂z[index] .= 0
        return (NoTangent(), ∂z, ∂x)
    end
    return force(z, x), force_pullback
end


function force(z::AbstractMatrix, x::AbstractMatrix, α::Float32)
    #Do not force with NaN values in observation
    index = findall(!isnan, x)
    z[index] = (1 - α) * z[index] + α * x[index]
    return z
end


function ChainRulesCore.rrule(
    ::typeof(force),
    z::AbstractMatrix,
    x::AbstractMatrix,
    α::Float32,
)
    function force_pullback(ΔΩ)
        index = findall(!isnan, x)
        ∂z = ΔΩ
        ∂x = similar(ΔΩ)
        ∂z[index] .= (1 .- α) .* ΔΩ[index]
        ∂x[index] .= (α .* ΔΩ[index])
        ∂α = (sum((x[index] .- z[index]) .* ΔΩ[index]))
        return (NoTangent(), ∂z, ∂x, ∂α)
    end
    return force(z, x, α), force_pullback
end
