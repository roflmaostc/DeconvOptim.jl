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

However, our approach to express the regularizers with [Tullio.jl](https://github.com/mcabbott/Tullio.jl) is currently not performant with GPUs.
Therefore, to use `CuArray`s with regularizers, you need to choose [`TV_cuda`](@ref).
Other regularizers are not yet supported since we hope that Tullio.jl will be one day mature enough to produce
reasonable fast gradients for CUDA kernels as well.
