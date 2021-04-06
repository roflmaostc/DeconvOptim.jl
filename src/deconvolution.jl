export deconvolution

"""
    deconvolution(measured, psf; <keyword arguments>)
Computes the deconvolution of `measured` and `psf`. Return parameter is a tuple with
two elements. The first entry is the deconvolved image. The second return parameter 
is the output of the optimization of Optim.jl

Multiple keyword arguments can be specified for different loss functions,
regularizers and mappings.

# Arguments
- `loss=Poisson()`: the loss function taking a vector the same shape as measured. 
- `regularizer=GR()`: A regularizer function, same form as `loss`.
- `λ=0.05`: A float indicating the total weighting of the regularizer with 
    respect to the global loss function
- `background=0`: A float indicating a background intensity level.
- `mapping=Non_negative()`: Applies a mapping of the optimizer weight. Default is a 
              parabola which achieves a non-negativity constraint.
- `iterations=20`: Specifies a number of iterations after the optimization.
    definitely should stop.
- `plan_fft=true`: Boolean whether plan_fft is used. Gives a slight speed improvement.
- `padding=0`: an float indicating the amount (fraction of the size in that dimension) 
        of padded regions around the reconstruction. Prevents wrap around effects of the FFT.
        A array with `size(arr)=(400, 400)` with `padding=0.05` would result in reconstruction size of 
        `(440, 440)`. However, we only return the reconstruction cropped to the original size.
        `padding=0` disables any padding.
- `optim_options=nothing`: Can be a options file required by Optim.jl. Will overwrite iterations.
- `optim_optimizer=LBFGS()`: The chosen Optim.jl optimizer. 


# Example
```julia-repl
julia> using DeconvOptim, TestImages, Colors, Noise;

julia> img = Float32.(testimage("resolution_test_512"));

julia> psf = Float32.(generate_psf(size(img), 30));

julia> img_b = conv_psf(img, psf);

julia> img_n = poisson(img_b, 300);

julia> @time res, o = deconvolution(img_n, psf);
```
"""
function deconvolution(measured::AbstractArray{T, N}, psf;
        loss=Poisson(),
        regularizer=GR(),
        λ=0.05,
        background=0,
        mapping=Non_negative(),
        iterations=20,
        plan_fft=true,
        padding=0.00,
        optim_options=nothing,
        optim_optimizer=LBFGS(),
        ) where {T, N}

    background = convert(eltype(measured), background) 


    # rec0 will be an array storing the final reconstruction
    # we choose it larger than the measured array to reduce
    # wrap around artifacts of the Fourier Transform
    # we create a array size_padded which stores a new array size
    # our reconstruction array will be larger than measured
    # to prevent wrap around artifacts
    size_padded = []
    for i = 1:ndims(measured)
        # if the size of the i-th dimension is 1
        # don't do any padding because there won't be no
        # convolution in that dimension
        if  size(measured)[i] == 1
            push!(size_padded, 1)
        else
            # only pad, if padding is true
            if ~iszero(padding)
                # 2 * ensures symmetric padding
                # minimum padding is 2 (4 in total) on each side
                x = max(4, 2 * round(Int, size(measured)[i] * padding))
            else
                x = 0
            end
            push!(size_padded, size(measured)[i] + x)
        end
    end


    # the dimensions we do the Fourier Transform over
    fft_dims = collect(1:ndims(psf)) 

    # we divide by the maximum to normalize
    rescaling = maximum(measured) 
    measured = measured ./ rescaling
    # create rec0 which will be the initial guess for the reconstruction
    rec0 = similar(measured, (size_padded)...)
    fill!(rec0, zero(eltype(measured))) 
    
    # alternative rec0_center, unused at the moment
    #rec0_center = m_invf(abs.(conv_psf(measured, psf, fft_dims)))
    #
    # take the mean as the initial guess
    # therefore has the same total energy at the initial guess as
    # measured
    one_arr = similar(measured, size(measured))
    fill!(one_arr, one(eltype(measured)))
    rec0_center = mean(measured) .* one_arr
    center_set!(rec0, rec0_center)


    # psf_n is the psf with the same size as rec0 but only in that dimensions
    # that were supported by the initial psf. Broadcasting of psf with less 
    # dimensions is still supported
    # we put the small psf into the new one
    # it is important to pad the PSF instead of the OTF
    
    psf_new_size = Array{Int}(undef, 0)
    for i = 1:ndims(psf)
        push!(psf_new_size, size(rec0)[i])
    end
    
    psf_new_size = tuple(psf_new_size...)
    psf_n = similar(rec0, psf_new_size)
    fill!(psf_n, zero(eltype(rec0)))
    psf_n = center_set!(psf_n, fftshift(psf))
    psf = ifftshift(psf_n)

    # the psf should be normalized to 1
    psf ./= sum(psf)

    # use plan_fft (function of the FFTW.jl has the same name)
    # for speed improvement
    if plan_fft
        # otf is obtained by rfft(psf)
        # therefore size(psf) != size(otf)
        otf, conv = plan_conv_r(psf, rec0, fft_dims) 
    else
        otf = rfft(psf, fft_dims)
        conv(rec, otf) = conv_otf_r(rec, otf, fft_dims)
    end
    

    # forward model is a convolution
    # due to numerics, we need to clip at 0
    # analytically it's a convolution psf ≥ 0 and image ≥ 0
    # so it must be conv(psf, image) ≥ 0
    forward(x) = center_extract((conv_aux(conv, x, otf)) .+ background, size(measured))
   
    # pass to more general optimization
    res_out, res = invert(measured, rec0, forward;
                          iterations=iterations, λ=λ,
                          regularizer=regularizer,
                          optim_optimizer=optim_optimizer,
                          optim_options=optim_options,
                          mapping=mapping,
                          loss=loss)

    res_out .*= rescaling
    # since we do some padding we need to extract the center part
    res_out = center_extract(res_out, size(measured))    
    return res_out, res
end


