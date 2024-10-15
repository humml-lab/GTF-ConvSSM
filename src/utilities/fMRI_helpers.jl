using Distributions: Gamma, pdf, Uniform, Normal
using DSP
using Wavelets
using Deconvolution
using Random


function spm_Gpdf(x, α, β)
    Θ = 1/β #just transforming to a different notation of the Gamma distribution
    dist = Gamma(α,Θ) #Gamma Distribution x^(α-1) e^(-x/θ)/(Γ(α)θ^α)
    return pdf.(dist, x)
end

function spm_hrf(RT)
    #Return a hemodynamic response function
    """
    RT   - scan repeat time
    p    - parameters of the response function (consisting of two Gamma functions)
                                                    defaults
                                                    (seconds)
    p(1) - delay of response (relative to onset)          6
    p(2) - delay of undershoot (relative to onset)       16
    p(3) - dispersion of response                         1
    p(4) - dispersion of undershoot                       1
    p(5) - ratio of response to undershoot                6
    p(6) - onset (seconds)                                0
    p(7) - length of kernel (seconds)                    32
    """
    p = [6., 16., 1., 1., 6., 0., 32.]
    
    u = (range(0, floor(Int64,p[7]/RT), step=1) |> collect)*RT .- p[6] #array of timepoints with distance RT fitting inside the kernel length
    
    hrf = spm_Gpdf(u, p[1]/p[3], 1/p[3]) - spm_Gpdf(u, p[2]/p[4], 1/p[4])/p[5] #hrf evaluated at time points u
    #convolution can not cope with different data types, so it have to be transformed from Float64 to Float32
    hrf = Float32.(hrf)
    return hrf/sum(hrf) #Normalizing to 1
end

function getConvolutionMatrix(impulse_response::AbstractVector, signal_len::Int)
    padding = length(impulse_response) - 1
    m = size(impulse_response)[1] #dimension of full hrf kernel
    n = signal_len #dimension of time points in the signal given (how long is the sequence of sequential observations (x_0...x_t) given?) Important to determine the maximal size of the
    # convolution matrix which has to map (x)_(1:t) = z_(1-(kernel_length+1) m:t)*Con => dim(Con) = (n+m-1, n). Any signal prior to 1-m-1 do not effect the observations anyway
    matrix = zeros((n+m-1, n)) #Initialize ConvolutionMatrix
    for i in 1:n
        matrix[i:(m+i-1), i] = impulse_response #insert the impulse response in every column shifted one row down
    end
    return Float32.(matrix[begin+padding:end-padding,:])
end

function hrf_conv(X::AbstractVecOrMat, hrf_signal::AbstractVector)
    padding = size(hrf_signal,1)-1
    X_conv = DSP.conv(X,hrf_signal)[1+padding:end-padding,:]
    return X_conv
end

function Wiener_Deconvolution(X::AbstractMatrix, hrf::AbstractVector, wiener_noise_level::Float32)
    dim = size(X,2)
    T = size(X,1)
    Te = (T%2!=0 ? T-1 : T)
    #noisest cannot handle uneven length
    hrf_padded = vcat(hrf,zeros(T-length(hrf))) #needed for the deconvolution
    X_deconv = zeros(size(X))
    low_noise_list = Int[]

    Threads.@threads for i in 1:dim
        signal = X[:,i]
        noise_level = noisest(signal[1:Te])
        #if the noise is too low the wiener_noise_level is chosen
        if noise_level < wiener_noise_level
            append!(low_noise_list, i)
            noise_level = wiener_noise_level
        end
        denoised_signal = (T%2!=0 ? denoise(cat(signal,[0], dims=1))[1:end-1] : denoise(signal))
        #Need normalization (sqrt(T)*noise_level)^2 to get the mean of the abs2fft(). Normalization 
        #with the length of the discrete fouriertransformed sequence sqrt(T)
        #and ^2 since we need the absolute square 
        X_deconv[:,i] = wiener(signal, denoised_signal, (sqrt(T)*noise_level)^2, hrf_padded)
    end
    return X_deconv, low_noise_list
end

function get_deconv_data(path::String, data_id::String, train_test_split::Int, cut_l::Float32, cut_r::Float32, min_conv_noise::Float32, hrf::AbstractVector, X_full::AbstractMatrix; device = cpu, dtype = Float32)
    X = X_full[1:train_test_split, :] #use as dataset only trainset
    X_test = X_full[train_test_split+1:end, :]

    h = length(hrf)
    #decides how many states at the boundary of the deconvoluted time series are set to NaN and not used
    cut_l = (cut_l<=1 ? Int(floor(cut_l*h)) : Int(cut_l))
    cut_r = (cut_r<=1 ? Int(floor(cut_r*h)) : Int(cut_r))

    #hrf_padded = vcat(hrf,zeros(train_test_split-length(hrf))) #needed for the deconvolution
    
    deconv_path = path_deconv(path, data_id, !isempty(X_test), train_test_split, cut_r, min_conv_noise)

    if isfile(deconv_path) #see if the deconv data was already created before and therefore does not need to be computed again
        X_deconv_full = npzread(deconv_path) .|> dtype |> device
        X_deconv = X_deconv_full[1:train_test_split, :] #use as dataset only trainset
        X_deconv_test = X_deconv_full[train_test_split+1:end, :]
        #set the values at the edges to NaN so they will not be used as initial conditions (left) or forcing signals (right)
        X_deconv[1:cut_l,:].= NaN; 
        X_deconv[end-cut_r+1:end,:] .= NaN  
    else
        #determine the data series which give convoluted X and R (<=> conv_mat * X_deconv = X), using Wiener deconvolution
        X_deconv, low_noise_list = Wiener_Deconvolution(X, hrf, min_conv_noise) 
        X_deconv = X_deconv .|> dtype
        isempty(low_noise_list) ? nothing : println("The computed noise_levels of dimensions $(low_noise_list) are lower than the chosen minimal noise level $(min_conv_noise).")
        X_deconv_test = similar(X_test) #To determine the initital states for the test error
        Threads.@threads for i in 1:size(X_test,1) #for the length of the test set 'shift the split', only include as much additional information as needed 
            X_deconv_loc, _ = Wiener_Deconvolution(X_full[1+i:train_test_split+i, :], hrf, min_conv_noise)
            X_deconv_test[i,:] = X_deconv_loc[train_test_split - cut_r,:] #do not use outermost deconv value but cut_l away from edge
        end
        X_deconv_full = cat(X_deconv, X_deconv_test; dims=1)
        npzwrite(deconv_path, X_deconv_full)
        X_deconv[1:cut_l,:].= NaN; 
        X_deconv[end-cut_r+1:end,:] .= NaN  #Due to the length of the convolution these values can not be determined and the Wiener deconvolution produces artifacts

    end
    return X_deconv, X_deconv_test 
end


#check whether a data_id is given which has to be consider in the path or not
function path_deconv(path::String, data_id::String, test_set::Bool, train_test_split::Int, cut_r::Int, min_conv_noise::Float32)
    if isempty(data_id) #no data_id, no folder structure. Just add Deconv... into file name
        path = chopsuffix(path, ".npy")
         #only if there is no test_set the train_test_split and the cutting of at the right edge changes X_deconv_full
        deconv_path = (!test_set ? path*"_Deconv_noise_$(min_conv_noise).npy" : path*"_Deconv_noise_$(min_conv_noise)_ttsplit_$(train_test_split)_cutting_$(cut_r).npy")
    else 
        path = (last(path)=='/' ? chop(path) : path)  #check if the last character in the path is / and chop it of to create the path for the deconv data
         #only if there is no test_set the train_test_split and the cutting of at the right edge changes X_deconv_full
        deconv_folder_path = (!test_set ? path*"_Deconv_noise_$(min_conv_noise)" : path*"_Deconv_noise_$(min_conv_noise)_ttsplit_$(train_test_split)_cutting_$(cut_r)")
        !ispath(deconv_folder_path) ? mkpath(deconv_folder_path) : nothing #check if the folder for the deconv data exists and created it if necessary
        deconv_path = joinpath(deconv_folder_path, string(data_id,".npy"))
    end
    return deconv_path
end


