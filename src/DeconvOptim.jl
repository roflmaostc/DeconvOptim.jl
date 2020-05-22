module DeconvOptim

using Zygote: @adjoint, gradient
using Optim
using Statistics
using FFTW

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
function deconvolution(measured, psf;
        lossf=Poisson(),
        regularizerf=GR(),
        mappingf=Non_negative(),
        iterations=50,
        options=nothing)

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
    lossf2(F, G, rec) = lossf(F, G, rec, otf, measured)

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

    rec0 = m_invf(conv_real_otf(measured, otf))
    res = Optim.optimize(Optim.only_fg!(f!), rec0, LBFGS(), options)

    return mf(Optim.minimizer(res)), res

end

# Module
end
