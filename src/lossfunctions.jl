export Poisson, poisson_aux
export Gauss, gauss_aux
export ScaledGauss, scaled_gauss_aux


"""
    poisson_aux(μ, meas, storage=copy(μ))

Calculates the Poisson loss for `μ` and `meas`.
`μ` can be of larger size than `meas`. In that case
we extract a centered region from `μ` of the same size as `meas`.
""" 
function poisson_aux(μ, meas, storage=copy(μ))
    # due to numerical errors, μ can be negative or 0
    if minimum(μ) <= 0
        μ .= μ .+ eps(maximum(μ)) .+ abs.(minimum(μ))
    end
    storage .= μ .- meas .* log.(μ)
    return sum(storage)
end


 # define custom gradient for speed-up
 # ChainRulesCore offers the possibility to define a backward AD rule
 # which can be used by several different AD systems
function ChainRulesCore.rrule(::typeof(poisson_aux), μ, meas, storage=copy(μ))
    Y = poisson_aux(μ, meas, storage)

    function poisson_aux_pullback(xbar)
        storage .= xbar .* (one(eltype(μ)) .- meas ./ μ)
        return zero(eltype(μ)), storage, zero(eltype(μ)), zero(eltype(storage)) 
    end

    return Y, poisson_aux_pullback
end


"""
    Poisson()

Returns a function to calculate Poisson loss
Check the help of `poisson_aux`.
"""
function Poisson()
      return poisson_aux
end



"""
    gauss_aux(μ, meas, storage=copy(μ))

Calculates the Gauss loss for `μ` and `meas`.
`μ` can be of larger size than `meas`. In that case
we extract a centered region from `μ` of the same size as `meas`.
""" 
function gauss_aux(μ, meas, storage=copy(μ))
    storage .= abs2.(μ - meas)
    return sum(storage)
end

 # define custom gradient for speed-up
function ChainRulesCore.rrule(::typeof(gauss_aux), μ, meas, storage=copy(μ))
    Y = gauss_aux(μ, meas) 
    function gauss_aux_pullback(xbar)
        return zero(eltype(μ)), 2 .* (μ - meas), zero(eltype(μ)), zero(eltype(μ)) 
    end
    return Y, gauss_aux_pullback
end


"""
    Gauss()

Returns a function to calculate Gauss loss.
Check the help of `gauss_aux`.
"""
function Gauss()
    return gauss_aux
end





"""
    scaled_gauss_aux(μ, meas)
Calculates the scaled Gauss loss for `μ` and `meas`.
`μ` can be of larger size than `meas`. In that case
we extract a centered region from `μ` of the same size as `meas`.
"""
function scaled_gauss_aux(μ, meas, storage=copy(μ))
    μ[μ .<= 1f-8] .= 1f-8
    storage .= 1/2f0 .* log.(μ) .+ (meas .- μ).^2 ./ (2 .* μ)
    return sum(storage)
end

 # define custom gradient for speed-up
function ChainRulesCore.rrule(::typeof(scaled_gauss_aux), μ, meas, storage=copy(μ))
    Y = scaled_gauss_aux(μ, meas) 
    function scaled_gauss_aux_pullback(xbar)
        ∇ = (μ + μ.^2 - meas.^2)./(2 .* μ.^2)
        ∇[μ .<= 1f-8] .= 0 
        return zero(eltype(μ)), ∇, zero(eltype(μ)) 
    end
    return Y, scaled_gauss_aux_pullback
end


"""
    ScaledGauss()

Returns a function to calculate scaled Gauss loss.
Check the help of `scaled_gauss_aux`.
"""
function ScaledGauss()
    return scaled_gauss_aux
end
