export Poisson
export Gauss
export poisson_aux

function poisson_aux(conv, rec, otf, meas, up_sampling)
    n = length(rec)
    μ = conv(rec, otf)
    μ = center_extract(μ, size(meas))
    print("lol")
    return 1 ./ n .* sum(μ .- meas .* log.(μ))
end

@adjoint poisson_aux(conv, rec, otf, meas, up_sampling) = begin
    n = length(rec)
    μ = conv(rec, otf)

    # we need to pad the measured array with the values of μ
    meas_new = copy(μ)
    meas_new = center_set!(meas_new, meas)
    
    # calculate the final gradient
    out_arr = 1 ./ n .* conv(1 .- meas_new ./ μ, conj(otf))
    # we need to return a 5-tuple because the function poisson_aux has 5 parameter 
    b = (c -> (-1, c .* out_arr, -1, -1, -1))
   
    # calculate finally the loss value 
    loss_value = poisson_aux(conv,rec, otf, meas, up_sampling)
    return (loss_value, b)
end

"""
    Poisson()

Returns a function to calculate poisson loss and gradient of it.

"""
function Poisson()
      return poisson_aux
end


"""
    Gauss()

Returns a function to calculate Gauss loss and gradient of it.
"""
function Gauss()

    function gauss_loss!(F, G, rec, otf, meas)
        n = length(rec)

        ν = conv(rec, otf) - meas

        if G != nothing
            G .= 2 * conv(ν, otf)

        end
        if F != nothing
            return sum(ν.^2)
        end
    end
    return gauss_loss!
end
