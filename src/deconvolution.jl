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
- `regularizer=nothing`: A regularizer function, same form as `loss`. 
    See `GR`, `TV`, `Tikhonov` and the help page for different regularizers.
- `λ=0.05`: A float indicating the total weighting of the regularizer with 
    respect to the global loss function
- `background=0`: A float indicating a background intensity level.
- `mapping=Non_negative()`: Applies a mapping of the optimizer weight. Default is a 
              parabola which achieves a non-negativity constraint.
- `iterations=nothing`: Specifies a number of iterations after the optimization.
    definitely should stop. By default 20 iterations will be selected by generic_invert.jl, 
    if `nothing` is provided.
- `conv_dims`: A tuple indicating over which dimensions the convolution should happen.
               per default `conv_dims=1:ndims(psf)`
- `plan_fft=true`: Boolean whether plan_fft is used. Gives a slight speed improvement.
- `padding=0`: an float indicating the amount (fraction of the size in that dimension) 
    of padded regions around the reconstruction. Prevents wrap around effects of the FFT.
    A array with `size(arr)=(400, 400)` with `padding=0.05` would result in reconstruction size of 
    `(440, 440)`. However, if padding is >= 0.0, we only return the reconstruction cropped to the original size.
    For negative paddings, the absolute value is used, but the result maintains the padded size.
    `padding=0` disables any padding.
- `opt_package=Opt_Optim`: decides which backend for the optimizer is used.
- `opt=LBFGS()`: The chosen optimizer which must fit to `opt_package` 
- `opt_options=nothing`: Can be a options file required by Optim.jl. Will overwrite iterations.
- `initial=mean(measured)`: defines a value (or array) with the initial guess. This will be pulled through the inverse mapping function
                     and extended with a mean value (if border regions are used).
- `debug_f=nothing`: A debug function which must take a single argument, the current reconstruction.

!!! note
    If you want to provide your PSF model, ensure that centered around the first entry of the array (`psf[1]`).
    You may need to use `ifftshift` for a PSF model or a measured PSF.

# Example
```julia-repl
julia> using DeconvOptim, TestImages, Colors, Noise;

julia> img = Float32.(testimage("resolution_test_512"));

julia> psf = Float32.(generate_psf(size(img), 30));

julia> img_b = conv(img, psf);

julia> img_n = poisson(img_b, 300);

julia> res, o = deconvolution(img_n, psf);
```
"""
function deconvolution(measured::AbstractArray{T,N}, psf;
    loss=Poisson(),
    regularizer=GR(),
    λ=T(0.05),
    background=zero(T),
    mapping=Non_negative(),
    iterations=nothing,
    conv_dims=ntuple(+, ndims(psf)),
    padding=0.00,
    opt_options=nothing,
    opt=LBFGS(linesearch=BackTracking()),
    initial=mean(measured),
    debug_f=nothing,
    opt_package=Opt_Optim) where {T,N}


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
        if size(measured)[i] == 1
            push!(size_padded, 1)
        else
            # only pad, if padding is true
            if ~iszero(padding)
                # 2 * ensures symmetric padding
                # minimum padding is 2 (4 in total) on each side
                x = next_fast_fft_size(max(4, 2 * round(Int, size(measured)[i] * abs(padding))))
            else
                x = 0
            end
            push!(size_padded, size(measured)[i] + x)
        end
    end

    # we divide by the mean to normalize
    rescaling = mean(measured)
    measured = measured ./ rescaling
    initial = initial ./ rescaling

    # create rec0 which will be the initial guess for the reconstruction
    rec0 = similar(measured, (size_padded)...)
    fill!(rec0, one(eltype(measured)))

    # alternative rec0_center, unused at the moment
    #rec0_center = m_invf(abs.(conv(measured, psf, conv_dims)))
    #
    # take the mean as the initial guess
    # therefore has the same total energy at the initial guess as
    # measured
    csize = isa(initial, AbstractArray) ? size(initial) : size(measured)
    one_arr = similar(measured, size(measured))
    fill!(one_arr, mean(measured))
    center_set!(rec0, one_arr .* initial)
    mf, mf_inv = get_mapping(mapping)
    rec0 = mf_inv(rec0)
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

    otf, conv_temp = plan_conv(rec0, psf, conv_dims)


    # forward model is a convolution
    # due to numerics, we need to clip at 0
    # analytically it's a convolution psf ≥ 0 and image ≥ 0
    # so it must be conv(psf, image) ≥ 0
    forward(x) =
        let
            if iszero(background)
                center_extract((conv_aux(conv_temp, x, otf)), size(measured))
            else
                center_extract((conv_aux(conv_temp, x, otf) .+ background), size(measured))
            end
        end
    # pass to more general optimization
    res_out, res = invert(measured, rec0, forward;
        iterations=iterations, λ=λ,
        regularizer=regularizer,
        opt=opt,
        opt_options=opt_options,
        mapping=mapping,
        loss=loss,
        debug_f=debug_f, opt_package=opt_package)

    res_out .*= rescaling
    # since we do some padding we need to extract the center part
    # for negative paddings, keep the large size.
    if padding > 0.0
        res_out = center_extract(res_out, size(measured))
    end
    return res_out, res
end


