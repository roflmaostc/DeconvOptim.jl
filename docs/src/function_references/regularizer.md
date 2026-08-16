# Regularizers

## CPU
```@docs
TV
Tikhonov
GR
TH
HS
```


## CUDA

The regularizers `TV()`, `GR()`, `TH()` and `Tikhonov()` automatically use the CUDA variants below when called on a `CuArray`.
All of them accept `num_dims=nothing`, which is also the default, in which case the number of dimensions is inferred from the array upon use.

```@docs
DeconOptim.TV_cuda
DeconOptim.GR_cuda
DeconOptim.TH_cuda
DeconOptim.Tikhonov_cuda
```