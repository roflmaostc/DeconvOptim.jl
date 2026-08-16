# CUDA
We also support [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl).

## Load
Before using a `CuArray` simply invoke. 
```julia
using CUDA
```
Our routines need as input array either only `Array`s or `CuArray`s. To get the deconvolution running, both the PSF and the measured 
array needs to be a `CuArray`.
See also [our 3D example here](https://github.com/roflmaostc/DeconvOptim.jl/blob/master/examples/cuda_3D.ipynb).


## Issues with Regularizers

Our CPU regularizers are expressed with [Tullio.jl](https://github.com/mcabbott/Tullio.jl), which is currently not performant (and partly unsupported) with GPUs.
Therefore [`TV()`](@ref), [`GR()`](@ref), [`TH()`](@ref), [`Tikhonov()`](@ref) and [`HS()`](@ref) automatically dispatch to GPU compliant view/broadcast based implementations when 
called on a `CuArray`. You can use the explicit variants [`DeconvOptim.TV_cuda`](@ref), [`DeconvOptim.GR_cuda`](@ref), [`DeconvOptim.TH_cuda`](@ref), [`DeconvOptim.Tikhonov_cuda`](@ref) and [`DeconvOptim.HS_cuda`](@ref), but this should not be necessary and is discouraged.

All regularizers (both the Tullio and the view/broadcast based variants) accept `num_dims=nothing` in which case the number of dimensions
is inferred from the array upon use. For [`GR()`](@ref)/[`TV()`](@ref)/[`Tikhonov()`](@ref) the CPU path then pre-compiles a `@tullio` kernel for each
dimension `1:NMAX` (`NMAX = 10`) and dispatches on `ndims(arr)`; arrays with more dimensions or `CuArray`s fall back to the
view/broadcast based kernels and automatically choose `num_dims`. 
All regularizers support `sum_dims` (the dimensions over which the regularizer is computed, with the rest summed over) and per-dimension `weights`
(matched positionally to `sum_dims`). [`Tikhonov()`](@ref)/[`DeconvOptim.Tikhonov_cuda`](@ref) additionally support all three `mode`s (`"laplace"`, `"spatial_grad_square"`, `"identity"`) and `step`.