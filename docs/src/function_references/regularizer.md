# Regularizers

## CPU
```@docs
TV
Tikhonov
GR
TH
HS
```

All regularizers are constructed via keyword arguments and return a function which takes the array to be regularized
and returns a scalar. They support:
- `sum_dims`: the array dimensions over which the regularizer is computed; the remaining dimensions only take part in the summation.
- `weights`: per-dimension weights, matched positionally to `sum_dims`.
- `num_dims`: the number of spatial dimensions (inferred from the array upon use when `nothing`).

[`HS()`](@ref) computes the Hessian over exactly two dimensions (see its docstring).


## CUDA

The regularizers `TV()`, `GR()`, `TH()`, `Tikhonov()` and `HS()` automatically use the CUDA variants below when called on a `CuArray`.
All of them accept `num_dims=nothing`, which is also the default, in which case the number of dimensions is inferred from the array upon use.
The CUDA variants support the same `sum_dims` and `weights` keyword arguments as their CPU counterparts.

```@docs
DeconvOptim.TV_cuda
DeconvOptim.GR_cuda
DeconvOptim.TH_cuda
DeconvOptim.Tikhonov_cuda
DeconvOptim.HS_cuda
```