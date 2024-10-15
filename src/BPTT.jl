module BPTT
using Reexport
using Statistics

include("utilities/Utilities.jl")
@reexport using .Utilities

include("datasets/Datasets.jl")
@reexport using .Datasets

include("models/ObservationModels.jl")
@reexport using .ObservationModels

include("models/PLRNNs.jl")
@reexport using .PLRNNs

include("measures/Measures.jl")
@reexport using .Measures

include("tf_training/TFTraining.jl")
@reexport using .TFTraining

# meta stuff
include("parsing.jl")
export parse_commandline, initialize_model, initialize_optimizer, get_device, argtable, initialize_observation_model

include("multitasking.jl")
export Argument, prepare_tasks, main_routine

end
