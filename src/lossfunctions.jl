
export Poisson
export Gauss



"""
    poisson_loss(img, otf, meas)
Computes the poisson loss between the `img` (filtered with the `otf`) with
respect to the measured image `meas`.
`img` and `meas` need to be in the same shape.
The `otf` needs to be generated out of the PSF by `rtff(psf)`.
"""


function Poisson()

    function f!(F, G, rec, otf, meas)
        n = length(rec)
        μ = conv_real_otf(rec, otf)

        if G != nothing
            G .= 1 ./ n .* conv_real_otf(1 .- meas ./ μ, conj(otf))
        end
        if F != nothing
            return 1 ./ n .* sum(μ .- meas .* log.(μ))
        end
    end
    return f!
end


function Gauss(; regularizer=nothing, mapping=nothing)

    if mapping == nothing
        mapping = IDm()
    end
    m, ∇m, m_inv = mapping

    if regularizer == nothing
        regularizer = ID()
    end
    reg, ∇reg = regularizer

    function f!(F, G, img, otf, meas)
        n = length(img)

        img_m = m(img)
        ν = conv_real_otf(img_m, otf) - meas

        if G != nothing
            G .= ∇m(img) .* (∇reg(img_m) .+
                 2 * conv_real_otf(ν, otf))

        end
        if F != nothing
            return sum(ν.^2)
        end
    end
    return f!, m, m_inv
end
