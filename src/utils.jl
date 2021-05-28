export generate_psf
export generate_downsample, my_interpolate 
export center_extract, center_set!, get_indices_around_center, center_pos

"""
    generate_downsample(num_dim, downsample_dims, factor)

Generate a function (based on Tullio.jl) which can be used to downsample arrays.
`num_dim` (Integer) are the dimensions of the array.
`downsample_dims` is a list of which dimensions should be downsampled.
`factor` is a downsampling factor. It needs to be an integer number.

# Examples
```jldoctest
julia> ds = generate_downsample(2, [1, 2], 2) 
[...]
julia> ds([1 2; 3 4; 5 6; 7 8])
2×1 Array{Float64,2}:
 2.5
 6.5

julia> ds = generate_downsample(2, [1], 2)
[...]
julia> ds([1 2; 3 5; 5 6; 7 8])
2×2 Array{Float64,2}:
 2.0  3.5
 6.0  7.0
```
"""
function generate_downsample(num_dim, downsample_dims, factor)
    @assert num_dim ≥ length(downsample_dims)
    # create unit cell with Cartesian Index 
    # dims_units containts every where a 1 where the downsampling should happen
    dims_units = zeros(Int, num_dim)
    # here we set which dimensions should be downsamples
    dims_units[downsample_dims] .= 1
    # the unit cell expressed in CartesianIndex
    one = CartesianIndex(dims_units...)


    # create a list of symbols 
    # these list represents the symbols to access the arrays
    ind = :i
    inds_out = map(1:num_dim) do di
        i = Symbol(ind, di)
    end

    
    # output list for the add commands
    add = []
    # via CartesianIndex we can loop over all rectangular neighbours
    # we loop only over the neighbours in the downsample_dims
    for n = one:one * factor
        # for each index calculate the offset to the neighbour
        inds = map(1:num_dim) do di
                i = Symbol(ind, di)
                if n[di] == 0
                    di = i
                else
                    expr = :($factor * $i)
                    diff = -factor + n[di]
                    di = :($expr + $diff)
                end
            end
        # push this single neighbour to add list
        push!(add, :(arr[$(inds...)]))
    end
    # combine the different parts and divide for averaging
    expr = [:(@tullio res[$(inds_out...)] := (+($(add...))) / $factor ^ $(length(downsample_dims)))]
    #= return expr =#
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
A function which provides two output indices `i1` and `i2`
where `i2 - i1 = i_out`
The indices are chosen in a way that the set `i1:i2`
cuts the interval `1:i_in` in a way that the center frequency
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
        return 1 + x, i_in - x - 1 
    elseif mod(i_in, 2) == 0 && mod(i_out, 2) == 1
        x = (i_in - (i_out - 1)) ÷ 2
        return 1 + x, i_in - (x - 1)
    end
end


"""
    center_extract(arr, new_size_array)
Extracts a center of an array. 
`new_size_array` must be list of sizes indicating the output
size of each dimension. Centered means that a center frequency
stays at the center position. Works for even and uneven.
If `length(new_size_array) < length(ndims(arr))` the remaining dimensions
are untouched and copied.
# Examples
```jldoctest
julia> DeconvOptim.center_extract([1 2; 3 4], [1]) 
1×2 Array{Int64,2}:
 3  4
julia> DeconvOptim.center_extract([1 2; 3 4], [1, 1])
1×1 Array{Int64,2}:
 4
julia> DeconvOptim.center_extract([1 2 3; 3 4 5; 6 7 8], [2 2])
2×2 Array{Int64,2}:
 1  2
 3  4
```
"""
function center_extract(arr::AbstractArray, new_size_array)
    if size(arr) == new_size_array
        return arr
    end
    
    new_size_array = collect(new_size_array)

    # we construct two lists
    # the reason is, that we don't change higher dimensions which are not 
    # specified in new_size_array
    out_indices1 = [get_indices_around_center(size(arr)[x], new_size_array[x]) 
                    for x = 1:length(new_size_array)]
    
    out_indices1 = [x[1]:x[2] for x = out_indices1]
    
    # out_indices2 contains just ranges covering the full size of each dimension
    out_indices2 = [1:size(arr)[i] for i = (1 + length(new_size_array)):ndims(arr)]
    return arr[out_indices1..., out_indices2...]
end

function ChainRulesCore.rrule(::typeof(center_extract), arr, new_size_array)
    new_arr = center_extract(arr, new_size_array)

    function aux_pullback(xbar)
        if size(arr) == new_size_array
            return zero(eltype(arr)), xbar, zero(eltype(arr)) 
        else
            ∇ = similar(arr, size(arr))
            fill!(∇, zero(eltype(arr)))
            o = similar(arr, new_size_array)
            fill!(o, one(eltype(arr)))
            o .*= xbar
            center_set!(∇, o)
            return zero(eltype(arr)), ∇, zero(eltype(arr)) 
        end
    end

    return new_arr, aux_pullback
end


"""
    center_set!(arr_large, arr_small)
Puts the `arr_small` central into `arr_large`.
The convention, where the center is, is the same as the definition
as for FFT based centered.
Function works both for even and uneven arrays.
# Examples
```jldoctest
julia> DeconvOptim.center_set!([1, 1, 1, 1, 1, 1], [5, 5, 5])
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
    arr_large[out_is...] .= arr_small
    
    return arr_large
end


"""
    center_pos(x)
Calculate the position of the center frequency.
Size of the array is `x`
# Examples
```jldoctest
julia> DeconvOptim.center_pos(3)
2
julia> DeconvOptim.center_pos(4)
3
```
"""
function center_pos(x::Integer)
    # integer division
    return div(x, 2) + 1
end


"""
    generate_psf(psf_size, radius)

Generation of an approximate 2D PSF.
`psf_size` is the output size of the PSF. The PSF will be centered
around the point [1, 1],
`radius` indicates the pupil diameter in pixel from which the PSF is generated.

# Examples
```julia-repl
julia> generate_psf([5, 5], 2)
5×5 Array{Float64,2}:
 0.36       0.104721    0.0152786    0.0152786    0.104721
 0.104721   0.0304627   0.00444444   0.00444444   0.0304627
 0.0152786  0.00444444  0.000648436  0.000648436  0.00444444
 0.0152786  0.00444444  0.000648436  0.000648436  0.00444444
 0.104721   0.0304627   0.00444444   0.00444444   0.0304627
```
"""
function generate_psf(psf_size, radius)
    mask = rr_2D(psf_size) .<= radius
    mask_ft = fft(mask)
    psf = abs2.(mask_ft)
    return psf ./ sum(psf)
end


function rr_3D(s)
    rarr = zeros((s...))
    for k = 1:s[3]
        for j = 1:s[2]
            for i = 1:s[1]
                rarr[i, j, k] = sqrt( (i-center_pos(s[1]))^2 + (j-center_pos(s[2]))^2 + (k-center_pos(s[3]))^2)
            end
        end
    end
    return rarr
end

"""
    rr_2D(s)

Generate a image with values being the distance to the center pixel.
`s` specifies the output size of the 2D array.

# Examples
```julia-repl
julia> DeconvOptim.rr_2D((6, 6))
6×6 Array{Float64,2}:
 4.24264  3.60555  3.16228  3.0  3.16228  3.60555
 3.60555  2.82843  2.23607  2.0  2.23607  2.82843
 3.16228  2.23607  1.41421  1.0  1.41421  2.23607
 3.0      2.0      1.0      0.0  1.0      2.0
 3.16228  2.23607  1.41421  1.0  1.41421  2.23607
 3.60555  2.82843  2.23607  2.0  2.23607  2.82843
```
"""
function rr_2D(s)
    rarr = zeros((s...)) 
    for j = 1:s[2]
        for i = 1:s[1]
               rarr[i, j] = sqrt( (i - center_pos(s[1]))^2 + (j - center_pos(s[2]))^2)
        end
    end
    return rarr
end



function get_mapping(mapping)
    return mapping[1], mapping[2]
end

function get_mapping(mapping::Nothing)
    return identity, identity
end


function get_regularizer(reg, etype)
    return reg 
end

function get_regularizer(reg::Nothing, etype)
    x -> zero(etype)
end
