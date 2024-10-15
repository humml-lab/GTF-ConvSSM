# A scalable generative model for dynamical system reconstruction from neuroimaging data [NeurIPS 2024 (poster)]

## 1 Setup and usage
The entire project is written in [Julia](https://julialang.org/) using the [Flux](https://fluxml.ai/Flux.jl/stable/) deep learning stack.
### 1.1 Installation
Install the package in a new Julia environment and open the package manager using the `"]"` key, and activate and instantiate the package:
```julia
julia> `"]"`
(@v1.11) pkg> activate .
(GTF-convSSM) pkg> instantiate
```
We recommend using the latest version of [Julia (>v1.9)](https://julialang.org/downloads/).

# 2. Running the code
## 2.1 Single runs
To start a single training, execute the `main.jl` file, where arguments can be passed via command line. For example, to train a 3D shallowPLRNN with 50 hidden units for 1000 epochs using 4 threads, while keeping all other training parameters at their default setting, call
```
$ julia -t4 --project main.jl --model shallowPLRNN --latent_dim 3 --hidden_dim 50 --epochs 1000
```
in your terminal of choice (bash/cmd). The [default settings](settings/defaults.json) can also be adjusted directly; one can then omit passing any arguments at the call site. The arguments are also listed in  in the [`argtable()`](src/parsing.jl) function.

## 2.2 Multiple runs + grid search
To run multiple trainings in parallel e.g. when grid searching hyperparameters, the `ubermain.jl` file is used. Currently, one has to adjust arguments which are supposed to differ from the [default settings](settings/defaults.json), and arguments that are supposed to be grid searched, in the `ubermain` function itself. This is as simple as adding an `Argument` to the `ArgVec` vector, which is passed the hyperparameter name (e.g. `latent_dim`), the desired value, and and identifier for discernibility and documentation purposes. If value is a vector of values, grid search for these hyperparameters is triggered. 
```Julia
function ubermain(n_runs::Int)
    # load defaults with correct data types
    defaults = parse_args([], argtable())

    # list arguments here
    args = BPTT.ArgVec([
        Argument("experiment", "Lorenz63-GS"),
        Argument("model", "PLRNN"),
        Argument("latent_dim", [10, 20, 30], "M"),
        Argument("lat_model_regularization", [0.01, 0.1], "reg")
    ])

    [...]
end
```
This will run a grid search over `latent_dim` and `lat_model_regularization` hyperparameter options using the `PLRNN`.

The identifier (e.g. `"M"` in the snippet above) is only mandatory for arguments subject to grid search. Once Arguments are specified, call the ubermain file with the desired number of parallel worker proccesses (+ amount of threads per worker) and the number of runs per task/setting, e.g.
```{.sh}
$ julia -t2 --project ubermain.jl -p 10 -r 5
```
will queue 5 runs for each setting and use 10 parallel workers with each 2 threads.

## 2.3 Evaluating models
Evaluating trained model is done via `evaluate.jl`. Here, the path to the (test) data, the model experiment directory, and the settings to be passed to the various metrics employed, have to be provided. The settings include:

- $D_{stsp}$ &rarr;  # of bins $k$ (`Int`) or GMM scaling $\sigma$ (`Float32`) 
- PSE &rarr; power spectrum smoothing $\sigma$ (`Float32`)
- PE &rarr; # step ahead predictions $n$ (`Int`)

The correct $D_{stsp}$ is determined via multiple dispatch (i.e. argument type dependent).

# Specifics
Latent/Dynamics model choices
- vanilla PLRNN &rarr; [`PLRNN`](src/models/vanilla_plrnn.jl)
- mean-centered PLRNN &rarr; [`mcPLRNN`](src/models/vanilla_plrnn.jl)
- shallow PLRNN &rarr; [`shallowPLRNN`](src/models/shallow_plrnn.jl)
- clipped shallow PLRNN &rarr; [`clippedShallowPLRNN`](src/models/shallow_plrnn.jl)
- Deep PLRNN &rarr; [`deepPLRNN`](src/models/deep_plrnn.jl)
- dendritic PLRNN &rarr; [`dendPLRNN`](src/models/dendritic_plrnn.jl)
- clipped dendritic PLRNN &rarr; [`clippedDendPLRNN`](src/models/dendritic_plrnn.jl)
- dendritic PLRNN full W for each basis &rarr; [`FCDendPLRNN`](src/models/dendritic_plrnn.jl)

Observation model choices
1. No artifacts path given:
- Identity mapping (i.e. no observation model) &rarr; [`Identity`](src/models/affine.jl)
- Affine mapping / linear observation model (i.e. no observation model) &rarr; [`Affine`](src/models/affine.jl)
2. Artifacts path given:
- Regressor observation model  &rarr; [`Regressor`](src/models/regressor.jl)

If you try to run the code with argument "Affine" and a "artifacts_path", the program will give you a warning and end. You must choose an accepted combination.

## 1. Teacher forcing settings / types of training
### 1.1 Identity TF
Given that $N \leq M$, identity TF is performed by setting the observation model to `"Identity"`. In identity TF, the forcing signal is the ground truth data itself. The choice between weak TF and sparse TF is then mediated in the following way:
- $\alpha$ (`weak_tf_alpha`) = 1 &rarr; sparse TF with parameter $\tau$ (`teacher_forcing_interval`)
- $\alpha$ < 1 &rarr; weak TF with set $\alpha$

When $N > M$, the observation model is equipped with an additional Matrix $L$, which is used to estimate initial states for the $M-N$ non-read out states.

### 1.2 Inversion TF
Inversion TF can be used with any invertible observation model and any setting of $N$,$M$. Teacher signals are computed by applying the (estimated) inverse of the observation model to the $N$-dimensional ground truth data.

Further, the following cases exist:
- $N \geq M$ &rarr; weak TF with hyperparameter $\alpha$
- $N < M$ and $\alpha = 1$ &rarr; sparse TF with partial forcing and hyperparameter $\tau$
- $N < M$ and $\alpha < 1$ &rarr; weak TF
  
*Note*: To run inversion TF as used [here](https://arxiv.org/abs/2110.07238), just set $\alpha = 1.0$ and the `sequence_length` $T$ equal to the desired forcing interval $\tau$, and the parameter `teacher_forcing_interval` > $\tau$ (e.g. `sequence_length` = $\tau = 30$, `teacher_forcing_interval` = 31).

## 2. Data Format
Data for the algorithm is expected to be a single trajectory in form of a $T \times N$ matrix (file format: `.npy`), where $T$ is the total number of time steps and $N$ is the data dimensionality. [Examples](example_data/) are provided.

### 2.1 Naming convention of the data paths
There are two possiblities to give the data paths (i.e. `path_to_data`, `path_to_inputs` and `path_to_artifacts`).

- If the argument `data_id` is empty, the program expects a full path to the numpy file, e.g. "example_data/lorenz.npy".

- If `data_id` is given, for example `data_id` = "lorenz", the program interprets paths as folders, i.e. in this example
`path_to_data` = "example_data".

The notation using the `data_id` can be rather helpful if one wants to start multiple runs with different datasets with plain data + external inputs and or nuisance artifacts. Giving full paths would lead to runs with plain data and external inputs and or nuisance artifacts of another dataset, i.e. mixing of data not belonging together.

To be able to use `data_id` in a sensible way, one must store the plain data as `data_id`.npy in one folder and the matching external inputs and or nuisance artifacts in another folder also as `data_id`.npy.

Example:
`path_to_data`: "numpy_data/Observation_data/"

`path_to_artifacts`: "numpy_data/Regressor_data/"

`data_id`: "example1"

would load the plain data from "numpy_data/Observation_data/example1.npy" and the corresponding artifacts data from "numpy_data/Regressor_data/example1.npy"


# Versions
- Julia 1.8.5
