module DeconvOptim

using Zygote: @adjoint, gradient
using DiffRules
using Optim
using Statistics
using FFTW
using Distributed
using Tullio
using Images
using Interpolations
using Requires

export deconvolution


# via require we can check whether CUDA is loaded
# to enable CUDA support simply load CUDA before load DeconvOptim
to_gpu_or_not(x) = x
function __init__()
    @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" begin
        print("CUDA support is enabled")
        @eval using CUDA
        @eval to_gpu_or_not(x) = CuArray(x)
    end
end
    

include("utils.jl")
include("regularizer.jl")
include("mappings.jl")
include("lossfunctions.jl")





"""
    deconvolution(measured, psf; <keyword arguments>)
Computes the deconvolution of `measured` and `psf`.

Multiple keyword arguments can be specified for different loss functions,
regularizers and mappings.

# Arguments
- `lossf`: the lossfunction taking a vector the same shape as measured. 
           Default is `Poisson()`.
- `regularizerf`: A regularizer function, same form as `lossf`. Default
                  is `GR()`. 
- `mappingf`: Applies a mapping of the optimizer weight. Default is a 
              parabola which achieves a non-negativity constraint.
- `iterations`: Specifies a number of iterations after the optimization
    definitely should stop.
- `options`: Can be a options file required by Optim.jl. 
    Will overwrite iterations.
- `plan_fft`: Boolean whether plan_fft is used. Default `true`.
- `up_sampling` enables up sampling of the reconstruction. Needs to be an
    integer number. Default is 1.
"""
function deconvolution(measured::AbstractArray{T, N}, psf;
        lossf=Poisson(),
        regularizerf=GR(),
        mappingf=Non_negative(),
        iterations=20,
        options=nothing,
        plan_fft=true,
        padding=true,
        up_sampling=1) where {T, N}

    # rec0 will be the array storing the final reconstruction
    # we choose it larger than the measured array to reduce
    # wrap around artifacts of the Fourier Transform
    # we create a array size_padded which stores a new array size
    # our reconstruction array will be larger than measured
    # to prevent wrap around artifacts
    size_padded = []
    for i = 1:5
        # if measured is smaller than i-th dimensional
        # simply add size 1 as this dimension 
        # if the size of the i-th dimension is 1
        # don't do any padding because there won't be no
        # convolution happening in that dimension
        if ndims(measured) < i || size(measured)[i] == 1
            push!(size_padded, 1)
        else
            # only pad, if padding is true
            if padding
                # either add a total of 20% to each dimension
                # or add 5, if 20% is just too few   
                x = max(5, round(Int, size(measured)[i] * 0.2))
            else
                x = 0
            end
            push!(size_padded, size(measured)[i] + x)
        end
    end
    # create rec0 which will be the initial guess for the reconstruction
    rec0 = ones(T, (size_padded...)) * mean(measured)




    # do some reshaping if the data is not provided
    # as a 5D array
    if (N == 2 && ndims(psf) == 2)|| (N == 3 && ndims(psf) == 3)
        new_size = (size(measured)..., ones(Integer, 5 - N)...)
        measured = reshape(measured, new_size)
        psf = reshape(psf, new_size)
        fft_dims = collect(1:N) # [1,2,3] or [1,2]
    elseif N == 4 && ndims(psf) == 3 
        measured = reshape(measured, (size(measured)..., 1))
        psf = reshape(psf, (size(psf)..., 1, 1))
        fft_dims = [1, 2, 3] 
    else                       
        throw("Such a combination of the dimensions of PSF and Image
               is not supported at the moment.")
    end


    # psf_n is the psf with the same size as rec0
    # we put the small psf into the new one
    # it is important to pad the PSF instead of the OTF
    psf_n = zeros(Float32, size(rec0))
    psf_n = center_set!(psf_n, fftshift(psf))
    psf = ifftshift(psf_n)

    # the psf should be normalized to 1
    psf ./= sum(psf)

    # use plan_fft (function of the FFTW.jl has the same name)
    # for speed improvement
    if plan_fft
        # otf is obtained by rfft(psf) and therefore 
        # size(psf)[1:3] != size(rec0)[1:3]
        otf, conv = plan_conv_r(psf, rec0, fft_dims) 
    else
        otf = rfft(psf, fft_dims)
        conv = conv_otf_r
    end
    
    # we are running into a world age counter problem
    # https://docs.julialang.org/en/v1/manual/methods/#Redefining-Methods
    if up_sampling != 1
        throw("up_sampling is not supported at the moment")
        downsample = generate_downsample(5, 2, up_sampling)
        # fix according to documentation
        #= Base.invokelatest(downsample) =#
    end
 
    # Get the mapping functions to achieve constraints
    # like non negativity
    if mappingf != nothing
        mf, m_invf = mappingf
    else
        mf, m_invf = identity, identity
    end

    # if no regularizer is provided, simply use x -> 0
    if regularizerf == nothing
        regularizerf = x -> 0 
    end

    # create the loss function which depends simply on the x
    # does Zygote recognizes that the derivatives of mf are reoccuring
    loss(x) = (poisson_aux(conv, mf(x), otf, measured, 1) + regularizerf(mf(x)))
    
    
    
    # this is the function which will be provided to Optimize
    # check Optims documentation for the purpose of F and Get
    # but simply speaking F is the loss value and G it's gradient
    # depending whether one of them is nothing, we skip some computations
    function f!(F, G, rec)
        if G != nothing
            G .= gradient(loss, rec)[1]
        end
        if F != nothing
            return loss(rec)
        end
    end

    # if not special options are given, just restrict iterations
    if options == nothing
        options = Optim.Options(iterations=iterations)
    end
    
    # do the optimization with LBGFS
    res = Optim.optimize(Optim.only_fg!(f!), rec0, LBFGS(), options)

    # since we do some padding we need to extract the core part
    # also apply the mapping
    res_out = center_extract(mf(Optim.minimizer(res)), size(measured))    
    return res_out, res
end

# end module
end
