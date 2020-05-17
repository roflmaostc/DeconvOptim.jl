module DeconvOptim

using Zygote: @adjoint, gradient
using Optim
using Statistics

export deconvolution

include("utils.jl")
include("regularizer.jl")
include("mappings.jl")
include("lossfunctions.jl")


function deconvolution(meas, otf;
        iterations=50,
        options=nothing,
        loss=Poisson(; regularizer=GR()))

    # if not special options are given, just restrict iterations
    if options == nothing
        optim_options = Optim.Options(iterations=iterations)
    end

    lossf!, m, m_inv = loss

    meas_m = meas ./ mean(meas)
    img0 = m_inv(conv_real_otf(meas_m, otf))

    f! = (F, G, img) -> lossf!(F, G, img, otf, meas_m)
    res = Optim.optimize(Optim.only_fg!(f!), img0, LBFGS(), optim_options)

    return mean(meas) .* m(Optim.minimizer(res)), res
end


end
