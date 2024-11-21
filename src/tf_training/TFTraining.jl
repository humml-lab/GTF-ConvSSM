module TFTraining

using Reexport
#Packages for the training
using Flux
using CUDA: @allowscalar
using BSON: @save
using DataStructures
using DataFrames
using SparseArrays

using ..Utilities
using ..Measures
using ..PLRNNs
using ..ObservationModels
using ..Datasets

export AbstractTFRecur, 
    TFRecur,
    WeakTFRecur,
    init_state!,
    force,
    train_!,
    mar_loss,
    AR_convergence_loss,
    regularize

include("forcing.jl")
include("tfrecur.jl")
include("regularization.jl")
include("progress.jl")
include("trainings/training_vanilla.jl")
include("trainings/training_ext_input.jl")
include("trainings/training_nuis_artifact.jl")
include("trainings/training_ext_input_nuis_artifact.jl")
include("trainings/training_vanilla_conv.jl")
include("trainings/training_ext_input_conv.jl")
include("trainings/training_nuis_artifact_conv.jl")
include("trainings/training_ext_input_nuis_artifact_conv.jl")
end
