"""
    normalized_positive_definite(M)

Build a positive definite matrix with maximum eigenvalue of 1.

RNN weight matrix initialized proposed by Talathi & Vartak (2016)
[ https://arxiv.org/abs/1511.03771 ].
"""
function normalized_positive_definite(M::Int)
    R = randn(Float32, M, M)
    K = R'R ./ M + I
    λ = maximum(abs.(eigvals(K)))
    return K ./ λ
end

"""

Build normalized He weight matrix as initial weights for deep neural network.
He weight initialization is especially suited for the use with the relu activation function.

He weight matrix as proposed by Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (2015)
[ https://arxiv.org/abs/1502.01852 ].

"""
function he_weights(
    prev_dimensions::Int,
    next_dimensions::Int;
    eltype::Type{T} = Float32,
) where {T <: AbstractFloat}
    d = Normal(μ = 0.0, σ = sqrt(2 / prev_dimensions))
    R = randn(d, eltype, prev_dimensions, next_dimensions)
    return R
end

function uniform_init(shape::Tuple; eltype::Type{T} = Float32) where {T <: AbstractFloat}
    @assert length(shape) < 3
    din = Float32(shape[end])
    r = 1 / √din
    return uniform(shape, -r, r)
end

"""
    uniform_threshold_init(shape, Dataset)

Return a Matrix of shape `shape` filled with values within the 
minimum and maximum extends of `D`.

Used to initialize basis thresholds `H` of the dendritic PLRNN.
"""
function uniform_threshold_init(shape::Tuple{Int, Int}, X::AbstractMatrix)
    # compute minima and maxima of dataset
    lo, hi = minimum(X), maximum(X)
    return uniform(shape, lo, hi)
end

function initialize_A_W_h(M::Int)
    AW = normalized_positive_definite(M)
    A, W = diag(AW), offdiagonal(AW)
    h = zeros(Float32, M)
    return A, W, h
end

function initialize_L(M::Int, N::Int)
    if M == N
        L = nothing
    else
        L = uniform_init((M - N, N))
    end
end