
export Poisson




"""
    poisson_loss(img, otf, meas)
Computes the poisson loss between the `img` (filtered with the `otf`) with
respect to the measured image `meas`.
`img` and `meas` need to be in the same shape.
The `otf` needs to be generated out of the PSF by `rtff(psf)`.
"""


function Poisson(; regularizer=GR(), mapping=Non_negative())
    m, ∇m, m_inv = mapping

    if regularizer == nothing
        regularizer = ID()
    end

    reg, ∇reg = regularizer

    function f!(F, G, img, otf, meas)
        n = length(img)

        img_m = m(img)
        μ = Utils.conv_real_otf(img_m, otf)

        if G != nothing
            G .= ∇m(img) .* (∇reg(img_m) .+
                 1 / n .* Utils.conv_real_otf(1 .- meas ./ μ, conj(otf)))
        end
        if F != nothing
            return  reg(img_m) + 1 / n * sum(μ .- meas .* log.(μ))
        end
    end
    return f!, m, m_inv
end


function gauss(img, otf, meas)
    ν = my_conv_otf(img, otf) - meas
    return sum(ν.^2)
end

function ∇gauss(img, otf, meas)
    ν = my_conv_otf(img, otf) - meas
    return 2 * my_conv_otf(ν, otf)
end

function gauss_comb!(F, G, img, otf, meas)
    ν = my_conv_otf(img, otf) - meas
    if G != nothing
        G .= 2 * my_conv_otf(ν, otf)
    end
    if F != nothing
        return sum(ν.^2)
    end
end
