module Datasets

using Flux: batch
using NPZ
using ..Utilities

export path_with_id,
    load_dataset,
    AbstractDataset,
    Dataset,
    ExternalInputsDataset,
    NuisanceArtifactsDataset,
    ExternalInputsNuisanceArtifactsDataset,
    DatasetConv,
    ExternalInputsDatasetConv,
    NuisanceArtifactsDatasetConv,
    ExternalInputsNuisanceArtifactsDatasetConv,
    sample_batch,
    sample_sequence


include("dataset_helpers.jl")
include("dataset_vanilla.jl")
include("dataset_ext_input.jl")
include("dataset_nuis_artifact.jl")
include("dataset_ext_input_nuis_artifact.jl")
include("dataset_vanilla_conv.jl")
include("dataset_ext_input_conv.jl")
include("dataset_nuis_artifact_conv.jl")
include("dataset_ext_input_nuis_artifact_conv.jl")

end