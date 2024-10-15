
#check whether a data_id is given which has to consider in the path or not
path_with_id(path::String, data_id::String) = isempty(data_id) ? path : joinpath(path, string(data_id,".npy"))

function load_dataset(args::AbstractDict; device = cpu)
    if Float32(args["TR"])==0.0 #Unconvolution/non fMRI datasets
        # check if external inputs are provided
        if !isempty(args["path_to_inputs"])
            if !isempty(args["path_to_artifacts"])
                println("Path to external inputs and nuisance artifacts provided, initializing ExternalInputsNuisanceArtifactsDataset.")
                D = ExternalInputsNuisanceArtifactsDataset(
                    args["path_to_data"],
                    args["path_to_inputs"],
                    args["path_to_artifacts"],
                    args["data_id"],
                    Float32(args["train_test_split"]);
                    device = device
                )
            else
            println("Path to external inputs but not nuisance artifacts provided, initializing ExternalInputsDataset.")
            D = ExternalInputsDataset(
                args["path_to_data"],
                args["path_to_inputs"],
                args["data_id"],
                Float32(args["train_test_split"]);
                device = device
            )
            end
        else
            if !isempty(args["path_to_artifacts"])
                println("Path to nuisance artifacts but not external inputs provided, initializing NuisanceArtifactsDataset.")
                D = NuisanceArtifactsDataset(
                    args["path_to_data"],
                    args["path_to_artifacts"],
                    args["data_id"],
                    Float32(args["train_test_split"]);
                    device = device
                )
            else
            println("No path to external inputs or nuisance artifacts provided, initializing vanilla Dataset.")
            D = Dataset(args["path_to_data"], args["data_id"],
            Float32(args["train_test_split"]);
            device = device)
            end
        end
    else #Convolution/fMRI datasets
        # check if external inputs are provided
        if !isempty(args["path_to_inputs"])
            if !isempty(args["path_to_artifacts"])
                println("Path to external inputs and nuisance artifacts provided, initializing ExternalInputsNuisanceArtifactsDatasetConv.")
                D = ExternalInputsNuisanceArtifactsDatasetConv(
                    args["path_to_data"],
                    args["path_to_inputs"],
                    args["path_to_artifacts"],
                    args["data_id"],
                    Float32(args["train_test_split"]),
                    Float32(args["TR"]),
                    Float32(args["cut_l"]),
                    Float32(args["cut_r"]),
                    Float32(args["min_conv_noise"]);
                    device = device
                )
            else
            println("Path to external inputs but not nuisance artifacts provided, initializing ExternalInputsDatasetConv.")
            D = ExternalInputsDatasetConv(
                args["path_to_data"],
                args["path_to_inputs"],
                args["data_id"],
                Float32(args["train_test_split"]),
                Float32(args["TR"]),
                Float32(args["cut_l"]),
                Float32(args["cut_r"]),
                Float32(args["min_conv_noise"]);
                device = device
            )
            end
        else
            if !isempty(args["path_to_artifacts"])
                println("Path to nuisance artifacts but not external inputs provided, initializing NuisanceArtifactsDatasetConv.")
                D = NuisanceArtifactsDatasetConv(
                    args["path_to_data"],
                    args["path_to_artifacts"],
                    args["data_id"],
                    Float32(args["train_test_split"]),
                    Float32(args["TR"]),
                    Float32(args["cut_l"]),
                    Float32(args["cut_r"]),
                    Float32(args["min_conv_noise"]);
                    device = device
                )
            else
            println("No path to external inputs or nuisance artifacts provided, initializing vanilla DatasetConv.")
            D = DatasetConv(
            args["path_to_data"],
            args["data_id"],
            Float32(args["train_test_split"]),                     
            Float32(args["TR"]),
            Float32(args["cut_l"]),
            Float32(args["cut_r"]),
            Float32(args["min_conv_noise"]);
            device = device)
            end
        end
        #check if sequence length is longer than the hrf signals
        sequence_length = args["sequence_length"]
        hrf_length = length(D.hrf)
        X_length = length(D.X)
        cutl = findfirst(!isnan, D.X_deconv[:,1])-1
        cutr = findfirst(!isnan, D.X_deconv[end:-1:1,1])-1
        @assert sequence_length >= hrf_length "The length of the sequences $(sequence_length) is shorter than the length of the hrf $(hrf_length). Choose a bigger one!"
        @assert sequence_length < X_length-cutl-1 "The length of the sequences $(sequence_length) is bigger than the length of the suitable initial conditions $(len_X-cut-1). Choose a smaller one!"
    end
    return D
end