module PLRNNs

using Flux, LinearAlgebra

using ..Utilities

export AbstractPLRNN,
    AbstractVanillaPLRNN,
    AbstractDendriticPLRNN,
    AbstractDeepPLRNN,
    AbstractShallowPLRNN,
    PLRNN,
    mcPLRNN,
    dendPLRNN,
    clippedDendPLRNN,
    FCDendPLRNN,
    shallowPLRNN,
    clippedShallowPLRNN,
    deepPLRNN,
    generate,
    jacobian,
    uniform_init

include("initialization.jl")
include("vanilla_plrnn.jl")
include("dendritic_plrnn.jl")
include("deep_plrnn.jl")
include("shallow_plrnn.jl")

end