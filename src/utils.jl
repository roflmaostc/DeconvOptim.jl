export generate_psf
export generate_downsample, my_interpolate 
export center_extract, center_set!, get_indices_around_center
export conv_psf, conv_otf, conv_otf_r, plan_conv_r



"""
    conv_psf(obj, psf [, dims])

Convolve `obj` with `psf` over `dims` dimensions.
Based on FFT convolution.
"""
function conv_psf(obj, psf, dims=[1, 2, 3])
    return real(ifft(fft(obj, dims) .* fft(psf, dims)))
end


"""
    conv_otf(obj, otf [, dims])

Performs a FFT-based convolution of an `obj`
with an `otf`. `otf = fft(psf)`. The 0 frequency of the `otf` must be located
at position [1, 1, 1].
The `obj` can be of arbitrary dimension but `ndims(psf) ≥ ndims(otf)`.
The convolution happens over the `dims` array. Any further dimensions are broadcasted.
Per default `dims = [1, 2, 3]`.
"""
function conv_otf(obj, otf, dims=[1, 2, 3])
    return real(ifft(fft(obj, dims) .* otf, dims))
end


"""
    conv_otf_r(obj, otf [, dims])

Performs a FFT-based convolution of an `obj`
with an `otf`.
Same arguments as `conv_otf` but with `obj` being real and `otf=rfft(psf)`.
"""
function conv_otf_r(obj, otf, dims=[1, 2, 3])
    return real(irfft(rfft(obj, dims) .* otf, size(obj)[1], dims))
end


"""
    plan_conv_r(psf [, dims])

Pre-plan an optimized convolution for array shaped like `psf` (based on pre-plan FFT) 
along the given dimenions `dims`.
`dims = [1, 2, 3]` per default.
The 0 frequency of `psf` must be located at [1, 1, 1].
We return first the `otf` (obtained by `rfft(psf))`.
The second return is the convolution function `conv`.
`conv` itself has two arguments. `conv(obj, otf)` where `obj` is the object and `otf` the otf.

This function achieves faster convolution than `conv_psf(obj, psf)`.
"""
function plan_conv_r(psf, rec0, dims=[1, 2, 3])
    # do the preplanning step
    P = plan_rfft(rec0, dims)
    rec0_fft = P * rec0 
    P_inv = plan_irfft(rec0_fft, size(psf)[1], dims)
    
    # obtain the otf by real based fft
    otf = rfft(psf, dims)
    # construct the efficient conv function
    conv(obj, otf) = real(P_inv * ((P * obj) .* otf))

    return otf, conv
end


"""
    generate_downsample(num_dim, factor)

Generate a Tullio statement which can be used to downsample arrays.
`num_dim` are the dimensions of the array
`factor` is a downsampling factor. It needs to be an integer number,

"""
function generate_downsample(num_dim, downsample_dims, factor)
    # create unit cell with Cartesian Index 
    one = oneunit(CartesianIndex(ones(Int, downsample_dims)...))
    # output list
    add = []
    ind = :i
    # create list of symbols for each dimension
    inds_out = map(1:num_dim) do di
        i = Symbol(ind, di)
    end
    # via CartesianIndex we can loop over all rectangular neighbours
    for n = one:one * factor
        # for each index calculate the offset
        inds = map(1:downsample_dims) do di
            i = Symbol(ind, di)
            expr = :($factor * $i)
            diff = n[di] - factor
            di = :($expr + $diff)
        end
        inds = [inds..., inds_out[downsample_dims+1:end]...]
        push!(add, :(arr[$(inds...)]))
    end
    # combine the different parts and divide for averaging
    expr = [:(@tullio res[$(inds_out...)] := (+($(add...))) / $factor ^ $num_dim)]
    # evaluate to function
    @eval f = arr -> ($(expr...))
    return f
end

"""
    my_interpolate(arr, size_n, [interp_type])

Interpolates `arr` to the sizes provided in `size_n`.
Therefore it holds `ndims(arr) == length(size_n)`.
`interp_type` specifies the interpolation type.
See Interpolations.jl for all options
"""
function my_interpolate(arr, size_n, interp_type=BSpline(Linear()))

    # we construct a arr which includes the interpolation
    # type for each dimension
    interp = []
    for s in size_n
        # if the outpute size is of the s-th dimension=1, 
        # do NoInterp
        if s == 1
            push!(interp, NoInterp())
        else
            push!(interp, interp_type)
        end
    end
    # prepare the interpolation
    arr_n = interpolate(arr, Tuple(interp))
   
    # interpolate introduces fractional indices 
    # via LinRange we access these fractional indices
    inds = []
    for d = 1:ndims(arr)
        push!(inds, LinRange(1, size(arr)[d], size_n[d]))
    end

    # return the new array sampled at the positions of inds
    # this accessing actually interpolates the data 
    return arr_n(inds...)
end


"""
    get_indices_around_center(i_in, i_out)

A function which provides two output indices i1 and i2
where i2 - i1 = i_out
The indices are choosen in a way that the set i1:i2
cuts the intervall 1:i_in in a way that the center frequency
stays at the center position.
Works for both odd and even indices

"""
function get_indices_around_center(i_in, i_out)
    if (mod(i_in, 2) == 0 && mod(i_out, 2) == 0 
     || mod(i_in, 2) == 1 && mod(i_out, 2) == 1) 
        x = (i_in - i_out) ÷ 2
        return 1 + x, i_in - x
    elseif mod(i_in, 2) == 1 && mod(i_out, 2) == 0
        x = (i_in - 1 - i_out) ÷ 2
        return 2 + x, i_in - x 
    elseif mod(i_in, 2) == 0 && mod(i_out, 2) == 1
        x = (i_in - (i_out - 1)) ÷ 2
        return 1 + x, i_in - (x - 1)
    end
end


"""
    center_extract(arr, new_size)

Extracts a center of an array. 
`new_size` must be list of sizes indicating the output
size of each dimension. Centered means that a center frequency
stays at the center position. Works for even and uneven.
If `length(new_size) < length(size(arr))` the remaining dimensions
are untouched and copied.

# Examples
```julia-repl
julia> center_extract([[1,2] [3, 4]], [1])
1×2 Array{Int64,2}:
 2  4

julia> center_extract([[1,2] [3, 4]], [1, 1])
1×1 Array{Int64,2}:
4
```
"""
function center_extract(arr, index_arrays)
    index_arrays = collect(index_arrays)
    out_indices1 = [get_indices_around_center(size(arr)[x], index_arrays[x]) 
                    for x = 1:length(index_arrays)]
    
    out_indices1 = [x[1]:x[2] for x = out_indices1]


    out_indices2 = map(eval, [Symbol(":") for i = (1 + size(index_arrays)[1]):ndims(arr)])
#    return view(arr, out_indices1..., out_indices2...)
    return arr[out_indices1..., out_indices2...]
end


"""
    center_set!(arr_large, arr_small)

Puts the `arr_small` central into `arr_large`.
The convention, where the center is, is the same as the definition
as for FFT based centered.
Function works both for even and uneven arrays.

# Examples
```julia-repl
julia> center_set!([1, 1, 1, 1, 1, 1], [5, 5, 5])
6-element Array{Int64,1}:
 1
 1
 5
 5
 5
 1
```
"""
function center_set!(arr_large, arr_small)
    out_is = []
    for i = 1:ndims(arr_large)
        a, b = get_indices_around_center(size(arr_large)[i], size(arr_small)[i])
        push!(out_is, a:b)
    end

    #rest = ones(Int, ndims(arr_large) - 3)
    arr_large[out_is...] = arr_small
    
    return arr_large
end



function rr(img)
    s = size(img)
    rarr = similar(img)
    for i = 1:s[1]
        for j = 1:s[2]
            for k = 1:s[3]
                rarr[i, j, k] = sqrt( (i-s[1] / 2)^2 + (j-s[2] / 2)^2 + (k- s[3] / 2)^2)
            end
        end
    end
    return rarr
end

function generate_psf(img, r)
    mask = rr(img) .< r
    mask_ft = ifft(ifftshift(mask))
    psf = abs2.(mask_ft)
    return psf ./ sum(psf)
end
