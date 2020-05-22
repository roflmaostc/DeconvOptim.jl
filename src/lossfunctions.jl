
export Poisson
export Gauss



"""
    Poisson()

Returns a function to calculate poisson loss and gradient of it.

"""
function Poisson()

    """
        poisson_loss!(F, G, rec, otf, meas)
    Computes the poisson loss between the `rec` (filtered with the `otf`) with
    respect to the measured image `meas`.

    `rec`, `otf` and `meas` need to be in the same shape.
    If `F` is unequal to `nothing`, the function will return the loss.
    If `G` is unequal to `nothing` (and a array), the gradient
    will be stored withing `G`.
    """
    function poisson_loss!(F, G, rec, otf, meas)
        n = length(rec)
        μ = conv_real_otf(rec, otf)

        if G != nothing
            G .= 1 ./ n .* conv_real_otf(1 .- meas ./ μ, conj(otf))
        end
        if F != nothing
            return 1 ./ n .* sum(μ .- meas .* log.(μ))
        end
    end
    return poisson_loss!
end


"""
    Gauss()

Returns a function to calculate Gauss loss and gradient of it.
"""
function Gauss()

    function gauss_loss!(F, G, rec, otf, meas)
        n = length(rec)

        ν = conv_real_otf(rec, otf) - meas

        if G != nothing
            G .= 2 * conv_real_otf(ν, otf)

        end
        if F != nothing
            return sum(ν.^2)
        end
    end
    return gauss_loss!
end
