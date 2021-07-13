export conv, plan_conv, conv_psf, plan_conv_psf


"""
    conv(u, v[, dims])
    Convolve `u` with `v` over `dims` dimensions with an FFT based method.
    Note, that this method introduces wrap-around artifacts without
    proper padding/windowing.
# Arguments
* `u` is an array in real space.
* `v` is the array to be convolved in real space as well.
* Per default `ntuple(+, min(N, M)))` means that we perform the convolution 
    over all dimensions of that array which has less dimensions. 
    If `dims` is an array with integers, we perform convolution 
    only over these dimensions. Eg. `dims=[1,3]` would perform the convolution
    over the first and third dimension. Second dimension is not convolved.
If `u` and `v` are both a real valued array we use `rfft` and hence
the output is real as well.
If either `u` or `v` is complex we use `fft` and output is hence complex.
 # Examples
1D with FFT over all dimensions. We choose `v` to be a delta peak.
Therefore convolution should act as identity.
```jldoctest
julia> u = [1 2 3 4 5]
1×5 Array{Int64,2}:
 1  2  3  4  5
julia> v = [0 0 1 0 0]
1×5 Array{Int64,2}:
 0  0  1  0  0
julia> conv(u, v)
1×5 Matrix{Float64}:
 4.0  5.0  1.0  2.0  3.0
```
2D with FFT with different `dims` arguments.
```jldoctest
julia> u = 1im .* [1 2 3; 4 5 6]
2×3 Matrix{Complex{Int64}}:
 0+1im  0+2im  0+3im
 0+4im  0+5im  0+6im
julia> v = [1im 0 0; 1im 0 0]
2×3 Matrix{Complex{Int64}}:
 0+1im  0+0im  0+0im
 0+1im  0+0im  0+0im
julia> conv(u, v)
2×3 Matrix{ComplexF64}:
 -5.0+0.0im  -7.0+0.0im  -9.0+0.0im
 -5.0+0.0im  -7.0+0.0im  -9.0+0.0im
```
"""
function conv(u::AbstractArray{T, N}, v::AbstractArray{D, M}, dims=ntuple(+, min(N, M))) where {T, D, N, M}
    return ifft(fft(u, dims) .* fft(v, dims), dims)
end

function conv(u::AbstractArray{<:Real, N}, v::AbstractArray{<:Real, M}, dims=ntuple(+, min(N, M))) where {N, M}
    return irfft(rfft(u, dims) .* rfft(v, dims), size(u, dims[1]), dims)
end

"""
    conv_psf(u, psf[, dims])
`conv_psf` is a shorthand for `conv(u,ifftshift(psf))`. For examples see `conv`.
"""
function conv_psf(u::AbstractArray{T, N}, psf::AbstractArray{D, M}, dims=ntuple(+, min(N, M))) where {T, D, N, M}
    return conv(u, ifftshift(psf, dims), dims)
end

 # define custom adjoint for conv
 # so far only defined for the derivative regarding the first component
function ChainRulesCore.rrule(::typeof(conv), u::AbstractArray{T, N}, v::AbstractArray{D, M},
                              dims=ntuple(+, min(N, M))) where {T, D, N, M}
    Y = conv(u, v, dims)
    function conv_pullback(barx)
        z = zero(eltype(u))
        return NoTangent(), conv(barx, conj(v), dims), z, z
    end 
    return Y, conv_pullback
end



"""
    plan_conv(u, v [, dims])
Pre-plan an optimized convolution for arrays shaped like `u` and `v` (based on pre-plan FFT)
along the given dimenions `dims`.
`dims = 1:ndims(u)` per default.
The 0 frequency of `u` must be located at the first entry.
We return two arguments: 
The first one is `v_ft` (obtained by `fft(v)` or `rfft(v)`).
The second return is the convolution function `pconv`.
`pconv` itself has two arguments. `pconv(u, v_ft=v_ft)` where `u` is the object and `v_ft` the v_ft.
This function achieves faster convolution than `conv(u, u)`.
Depending whether `u` is real or complex we do `fft`s or `rfft`s
# Examples
```jldoctest
julia> u = [1 2 3 4 5]
1×5 Matrix{Int64}:
 1  2  3  4  5
julia> v = [1 0 0 0 0]
1×5 Matrix{Int64}:
 1  0  0  0  0
julia> v_ft, pconv = plan_conv(u, v);
julia> pconv(u, v_ft)
1×5 Matrix{Float64}:
 1.0  2.0  3.0  4.0  5.0
julia> pconv(u)
1×5 Matrix{Float64}:
 1.0  2.0  3.0  4.0  5.0
```
"""
function plan_conv(u::AbstractArray{T, N}, v::AbstractArray{T, M}, dims=ntuple(+, N)) where {T, N, M}
    plan = get_plan(T)
    # do the preplanning step
    P = plan(u, dims)
    u_ft_stor = P * u
    P_inv = inv(P)

    v_ft = fft_or_rfft(T)(v, dims)
    out = copy(u)
    # construct the efficient conv function
    # P and P_inv can be understood like matrices
    # but their computation is fast
    conv(u, v_ft=v_ft) = p_conv_aux!(P, P_inv, u, v_ft, u_ft_stor, out)
    return v_ft, conv
end

"""
    plan_conv_psf(u, psf [, dims]) where {T, N}
`plan_conv_psf` is a shorthand for `plan_conv(u, ifftshift(psf))`. For examples see `plan_conv`.
"""
function plan_conv_psf(u::AbstractArray{T, N}, psf::AbstractArray{T, M}, dims=ntuple(+, N)) where {T, N, M}
    return plan_conv(u, ifftshift(psf, dims), dims)
end

function p_conv_aux!(P, P_inv, u, v_ft, u_ft_stor, out)
    #return P_inv.scale .* (P_inv.p * ((P * u) .* v_ft))  
    mul!(u_ft_stor, P, u)
    u_ft_stor .*= v_ft
    mul!(out, P_inv.p, u_ft_stor)
    out .*= P_inv.scale
    return out 
end

function ChainRulesCore.rrule(::typeof(p_conv_aux!), P, P_inv, u, v, u_ft_stor, out)
    Y = p_conv_aux!(P, P_inv, u, v, u_ft_stor, out)
    function conv_pullback(barx)
        z = zero(eltype(u))
        conj_v = let
            if eltype(v) <: Real
                v
            else
                conj(v)
            end
        end
        ∇ = p_conv_aux!(P, P_inv, collect(eltype(u).(barx)), conj_v, u_ft_stor, out)
        return NoTangent(), z, z, ∇, z, z, z
    end 
    return Y, conv_pullback
end

"""
    fft_or_rfft(T)
Small helper function to decide whether a real
or a complex valued FFT is appropriate.
"""
function fft_or_rfft(::Type{<:Real})
    return rfft
end

function fft_or_rfft(::Type{T}) where T
    return fft
end


"""
    get_plan(T)
Small helper function to decide whether a real
or a complex valued FFT plan is appropriate.
"""
function get_plan(::Type{<:Real})
    return plan_rfft
end

function get_plan(::Type{T}) where T
    return plan_fft
end
