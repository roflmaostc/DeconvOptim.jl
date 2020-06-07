module DeconvOptim

using Zygote: @adjoint, gradient
using ReverseDiff
using Optim
using Statistics
using FFTW
using Distributed
using Tullio

export deconvolution


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

"""
function deconvolution(measured::AbstractArray{T, N}, psf;
        lossf=Poisson(),
        regularizerf=GR(),
        mappingf=Non_negative(),
        iterations=50,
        options=nothing,
        plan_fft=true) where {T, N}

    if N == 2
        new_size = (size(measured)..., 1, 1, 1)
        measured = reshape(measured, new_size)
        psf = reshape(psf, new_size) 
    elseif N == 3 
        new_size = (size(measured)..., 1, 1)
        measured = reshape(measured, new_size)
        psf = reshape(psf, new_size) 
    end

    if plan_fft
        P = plan_rfft(measured, [1, 2, 3])
        measured_fft = P * measured
        P_inv = plan_irfft(measured_fft, size(measured)[1], [1, 2, 3])
        conv(rec, otf) = conv_real_otf_p(P, P_inv, rec, otf)
    else
        conv = conv_real_otf
    end

 
    # parsing of arguments
    if mappingf != nothing
        mf, ∇mf, m_invf = mappingf
    else
        mf, ∇mf, m_invf = identity, identity, identity
    end

    if regularizerf == nothing
        function n_regularizerf(F, G, rec)
            if G != nothing
                G = 0
            end
            if F != nothing
                return 0
            end
        end
        regularizerf = n_regularizerf
    end

    # transform the PSF to OTF, is more efficient
    otf = rfft(psf)

    # lossf needs otf and measured, but only once, therefore
    # an extra anonymous function
    lossf2(F, G, rec) = lossf(conv, F, G, rec, otf, measured)

    # this is the function which will be provided to Optimize
    # we use the trick and share computations
    function f!(F, G, rec)
        # map the reconstruction according to a function to a map function
        # can be a mapping to positive numbers for example
        m_rec = mf(rec)


        # if gradient should be calculated, we must allocate some space
        # for the gradient of the regularizer
        if G != nothing
            ∇reg = zeros(size(measured))
        end

        # calculate the regularizer loss
        reg = regularizerf(F, ∇reg, m_rec)
        # calculate the loss
        loss = lossf2(F, G, m_rec)

        if G != nothing
            G .= ∇mf(rec) .* (G .+ ∇reg)
        end
        if F != nothing
            return (loss .+ reg)
        end
    end

    # if not special options are given, just restrict iterations
    if options == nothing
        options = Optim.Options(iterations=iterations)
    end

    rec0 = m_invf(conv(measured, otf))
    res = Optim.optimize(Optim.only_fg!(f!), rec0, LBFGS(), options)

    return mf(Optim.minimizer(res)), res, rec0

end

# Module
end
