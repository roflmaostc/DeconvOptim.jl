module DeconvOptim

using Zygote: @adjoint, gradient
using Optim
using Statistics

export deconvolution, Deconvolution
include("utils.jl")
include("regularizer.jl")
include("mappings.jl")
include("lossfunctions.jl")


#function deconvolution(meas, otf;
#        iterations=50,
#        options=nothing,
#        loss=Poisson(; regularizer=GR()))
#
#    # if not special options are given, just restrict iterations
#    if options == nothing
#        optim_options = Optim.Options(iterations=iterations)
#    end
#
#    lossf!, m, m_inv = loss
#
#    img0 = m_inv(conv_real_otf(meas, otf))
#
#    f! = (F, G, img) -> lossf!(F, G, img, otf, meas)
#    res = Optim.optimize(Optim.only_fg!(f!), img0, LBFGS(), optim_options)
#
#    return m(Optim.minimizer(res)), res
#end


function deconvolution(meas, otf;
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

    # lossf needs otf and meas, but only once, therefore
    # an extra anonymous function
    lossf2(F, G, rec) = lossf(F, G, rec, otf, meas)

    # this is the function which will be provided to Optimize
    # we use the trick and share computations
    function f!(F, G, rec)
        # map the reconstruction according to a function to a map function
        # can be a mapping to positive numbers for example
        m_rec = mf(rec)


        # if gradient should be calculated, we must allocate some space
        # for the gradient of the regularizer
        if G != nothing
            ∇reg = zeros(size(meas))
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
        optim_options = Optim.Options(iterations=iterations)
    end

    rec0 = m_invf(conv_real_otf(meas, otf))
    res = Optim.optimize(Optim.only_fg!(f!), rec0, LBFGS(), optim_options)

    return mf(Optim.minimizer(res)), res

end

# Module
end
