module DeconvOptim

 # for optimization
using Optim
 #mean
using Statistics
using FFTW
 # for fast array regularizers
using Tullio
 # possible up_sampling 
using Interpolations
 # to check wether CUDA is enabled
using Requires
# for defining custom derivatives
using ChainRulesCore



export deconvolution


# via require we can check whether CUDA is loaded
# to enable CUDA support simply load CUDA before load DeconvOptim
 #to_gpu_or_not(x) = x
 #function __init__()
 #    @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" begin
 #        print("CUDA support is enabled")
 #        @eval using CUDA
 #        @eval to_gpu_or_not(x) = CuArray(x)
 #    end
 #end
    

include("forward_models.jl")
include("lossfunctions.jl")
include("mappings.jl")
include("regularizer.jl")
include("utils.jl")


# refresh Zygote to load the custom rrules defined with ChainRulesCore
using Zygote: @adjoint, gradient



"""
    deconvolution(measured, psf; <keyword arguments>)
Computes the deconvolution of `measured` and `psf`.

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
- `plan_fft=true`: Boolean whether plan_fft is used
- `padding=0`: an float indicating the amount (fraction of the size in that dimension) 
        of padded regions around the reconstruction. Prevents wrap around effects of the FFT.
        A array with `size(arr)=(400, 400)` with `padding=0.05` would result in reconstruction size of 
        `(440, 440)`. However, we only return the reconstruction cropped to the original size.
        `padding=0` disables any padding.
- `optim_options=nothing`: Can be a options file required by Optim.jl. Will overwrite iterations.
- `optim_optimizer=LBFGS()`: The choosen Optim.jl optimizer. 
"""
function deconvolution(measured::AbstractArray{T, N}, psf;
        loss=Poisson(),
        regularizer=GR(),
        λ=0.01,
        background=0,
        mapping=Non_negative(),
        iterations=20,
        plan_fft=true,
        padding=0.00,
        optim_options=nothing,
        optim_optimizer=LBFGS(),
        ) where {T, N}

    # do some type conversion to ensure same type everywhere
    # provides speed-up
    λ = convert(eltype(measured), λ)
    psf = convert(Array{eltype(measured)}, psf)
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
            if ~(padding ≈ 0)
                # 2 * ensures symmetric padding
                # minimum padding is 2 (4 in total) on each side
                x = max(4, 2 * round(Int, size(measured)[i] * padding))
            else
                x = 0
            end
            push!(size_padded, size(measured)[i] + x)
        end
    end
    # create rec0 which will be the initial guess for the reconstruction
    rec0 = ones(T, (size_padded...))

    # the dimensions we do the Fourier Transform over
    fft_dims = collect(1:ndims(psf)) 

    # psf_n is the psf with the same size as rec0
    # we put the small psf into the new one
    # it is important to pad the PSF instead of the OTF
    psf_n = zeros(eltype(rec0), size(rec0))
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
    

    # Get the mapping functions to achieve constraints
    # like non negativity
    if mapping != nothing
        mf, m_invf = mapping
    end


    # forward model is a convolution
    # due to numerics, we need to clip at 0
    # analytically it's a convolution psf ≥ 0 and image ≥ 0
    # so it must be conv(psf, image) ≥ 0
    forward(x) = (conv_aux(conv, x, otf)) .+ background
    # create the loss function which depends simply on the current rec  
    function total_loss(rec)
        # handle if there is a provided mapping function
        if mapping != nothing
            mf_rec = mf(rec)
        else
            mf_rec = rec
        end
        forward_v = forward(mf_rec)
        loss_v = loss(forward_v, measured)
        # handle if there is a regularizer
        if regularizer != nothing
            reg_v = regularizer(mf_rec)
            out = loss_v + λ * reg_v
        else
            out = loss_v
        end
        return out 
    end

    
    # this is the function which will be provided to Optimize
    # check Optims documentation for the purpose of F and Get
    # but simply speaking F is the loss value and G it's gradient
    # depending whether one of them is nothing, we skip some computations
    # we need to call Base.invokelatest becauser regualarizer is f function
    # generated at runtime with eval.
    # This leads to the common "world age problem" in Julia
    # for more details on that check:
    # https://discourse.julialang.org/t/dynamically-create-a-function-initial-idea-with-eval-failed-due-to-world-age-issue/49139/17
    function f!(F, G, rec)
        if G != nothing
            G .= Base.invokelatest(gradient, total_loss, rec)[1]
        end
        if F != nothing
            return Base.invokelatest(total_loss, rec)
        end
    end

    # if not special options are given, just restrict iterations
    if optim_options == nothing
        optim_options = Optim.Options(iterations=iterations)
    end
    
    # do the optimization with LBGFS
    res = Optim.optimize(Optim.only_fg!(f!), rec0, optim_optimizer, optim_options)

    # apply the mapping to get the real result
    if mapping != nothing
        res_out = mf(Optim.minimizer(res))
    else
        res_out = Optim.minimizer(res)
    end

    # since we do some padding we need to extract the center part
    res_out = center_extract(res_out, size(measured))    
    return res_out, res
end

# end module
end
