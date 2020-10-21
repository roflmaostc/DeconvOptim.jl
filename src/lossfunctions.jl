export Poisson
export Gauss
export poisson_aux

function poisson_aux(μ, meas)
    n = length(μ)
    μ = center_extract(μ, size(meas))
    return 1 ./ n .* sum(μ .- meas .* log.(μ))
end


function ChainRulesCore.rrule(::typeof(poisson_aux), μ, meas)
    n = length(μ)
    Y = poisson_aux(μ, meas)

    function poisson_aux_pullback(xbar)
        meas_new = copy(μ)
        meas_new = center_set!(meas_new, meas)
        return NO_FIELDS, xbar ./ n .* (1 .- meas_new ./ μ), DoesNotExist()
    end

    return Y, poisson_aux_pullback
end


 #@adjoint poisson_aux(μ, meas) = begin
 #    n = length(μ)
 #
 #    # we need to pad the measured array with the values of μ
 #    #= meas_new = zeros(eltype(meas), size(μ))#copy(μ) =#
 #    meas_new = copy(μ)
 #    meas_new = center_set!(meas_new, meas)
 #    
 #    # we need to return a 2-tuple because the function poisson_aux
 #    # has 2 parameter 
 #    ∇ = (c -> (c ./ n .* (1 .- meas_new ./ μ) , -1))
 #   
 #    # calculate finally the loss value 
 #    loss_value = poisson_aux(μ, meas)
 #    return (loss_value, ∇)
 #end

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
