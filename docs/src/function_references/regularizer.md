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
All of them accept `num_dims=nothing`, in which case the number of dimensions is inferred from the array upon use.

```@docs
TV_cuda
GR_cuda
TH_cuda
Tikhonov_cuda
```