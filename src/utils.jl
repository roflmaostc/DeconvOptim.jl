export generate_psf, conv_real_otf, conv_real
export generate_downsample
export my_interpolate 

export conv_real_otf5D


function conv_real_otf(rec, otf)
    return real(irfft(rfft(rec) .* otf, size(rec)[1]))
end

function conv_real_otf5D(rec, otf)
    return real(irfft(rfft(rec, [1, 2, 3]) .* otf, size(rec)[1], [1, 2, 3]))
end

function conv_real(img, psf)
    return real(irfft(rfft(rec, [1, 2, 3]) .* rfft(psf, [1, 2, 3]),
                      size(rec)[1], [1, 2, 3]))
end

function conv_real_otf_p(P, P_inv, rec, otf)
    return real(P_inv * ((P * rec) .* otf))
end

"""
    generate_downsample(num_dim, factor)

Generate a Tullio statement which can be used to downsample arrays.
`num_dim` are the dimensions of the array
`factor` is a downsampling factor. It needs to be an integer number,

"""
function generate_downsample(num_dim, factor)
    # create unit cell with Cartesian Index 
    one = oneunit(CartesianIndex(ones(Int, num_dim)...))
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
        inds = map(1:num_dim) do di
            i = Symbol(ind, di)
            expr = :($factor * $i)
            diff = n[di] - factor
            di = :($expr + $diff)
        end
        push!(add, :(arr[$(inds...)]))
    end
    # combine the different parts and divide for averaging
    expr = [:(@tullio res[$(inds_out...)] := (+($(add...))) / $factor ^ $num_dim)]
    # evaluate to function
    @eval f = arr -> ($(expr...))
    return f
end

function my_interpolate(arr, size_n)

    interp = []
    for s in size_n
        if s == 1
            push!(interp, NoInterp())
        else
            push!(interp, BSpline(Linear()))
        end
    end
    arr_n = interpolate(arr, Tuple(interp))
    
    inds = []
    for d = 1:ndims(arr)
        push!(inds, LinRange(1, size(arr)[d], size_n[d]))
    end

    return arr_n(inds...)
end


function rr(img)
    s = size(img)
    rarr = similar(img)
    for i = 1:s[1]
        for j = 1:s[2]
            rarr[i, j] = sqrt( (i-s[1] / 2)^2 + (j-s[2] / 2)^2)
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
