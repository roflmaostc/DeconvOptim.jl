export Poisson, poisson_aux
export Gauss, gauss_aux
export ScaledGauss, scaled_gauss_aux
export Anscombe, anscombe_aux

"""
    poisson_aux(μ, meas, storage=similar(μ))

Calculates the Poisson loss for `μ` and `meas`.
`μ` can be of larger size than `meas`. In that case
we extract a centered region from `μ` of the same size as `meas`.
""" 
function poisson_aux(μ, meas, storage=similar(μ))
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
function ChainRulesCore.rrule(::typeof(poisson_aux), μ, meas, storage)
    Y = poisson_aux(μ, meas, storage)

    function poisson_aux_pullback(xbar)
        storage .= xbar .* (one(eltype(μ)) .- meas ./ μ)
        return NoTangent(), storage, (ChainRulesCore.@not_implemented "Save computation"), (ChainRulesCore.@not_implemented "Save computation") 
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
    gauss_aux(μ, meas, storage=similar(μ))

Calculates the Gauss loss for `μ` and `meas`.
`μ` can be of larger size than `meas`. In that case
we extract a centered region from `μ` of the same size as `meas`.
""" 
function gauss_aux(μ, meas, storage=similar(μ))
    storage .= abs2.(μ - meas)
    return sum(storage)
end

 # define custom gradient for speed-up
function ChainRulesCore.rrule(::typeof(gauss_aux), μ, meas, storage)
    Y = gauss_aux(μ, meas) 
    function gauss_aux_pullback(xbar)
        return NoTangent(), 2 .* xbar .* (μ - meas), (ChainRulesCore.@not_implemented "Save computation"), (ChainRulesCore.@not_implemented "Save computation") 
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
    scaled_gauss_aux(μ, meas, storage=similar(μ); read_var=0)
Calculates the scaled Gauss loss for `μ` and `meas`.
`read_var=0` is the readout noise variance of the sensor.
`μ` can be of larger size than `meas`. In that case
we extract a centered region from `μ` of the same size as `meas`.
"""
function scaled_gauss_aux(μ, meas, storage=similar(μ); read_var=0)
    μ[μ .<= 1f-8] .= 1f-8
    storage .= log.(μ .+ read_var) .+ (meas .- μ).^2 ./ ((μ .+ read_var))
    return sum(storage)
end

 # define custom gradient for speed-up
function ChainRulesCore.rrule(::typeof(scaled_gauss_aux), μ, meas, storage; read_var=0)
    Y = scaled_gauss_aux(μ, meas) 
    function scaled_gauss_aux_pullback(xbar)
        ∇ = xbar .* (μ.^2 .- meas.^2 .+ μ .+ read_var.*(1 .- 2 .* (meas .- µ)))./((μ .+read_var).^2)
        ∇[μ .<= 1f-8] .= 0 
        return NoTangent(), ∇, (ChainRulesCore.@not_implemented "Save computation"), (ChainRulesCore.@not_implemented "Save computation"), (ChainRulesCore.@not_implemented "Save computation")
    end
    return Y, scaled_gauss_aux_pullback
end


"""
    ScaledGauss()

Returns a function to calculate scaled Gauss loss.
Check the help of `scaled_gauss_aux`.
"""
function ScaledGauss(read_var=0)
    return (µ, meas, storage=similar(µ)) -> scaled_gauss_aux(µ, meas, storage, read_var=read_var)
end





"""
    anscombe_aux(μ, meas, storage=similar(μ); b=1)

Calculates the Poisson loss using the Anscombe-based norm for `μ` and `meas`.
`μ` can be of larger size than `meas`. In that case
we extract a centered region from `μ` of the same size as `meas`.
`b=1` is the optional parameter under the `√`.
"""
function anscombe_aux(μ, meas, storage=similar(μ); b=1)
    # due to numerical errors, μ can be negative or 0
    mm = minimum(μ)
    if mm <= 0
        μ .= μ .+ eps(maximum(μ)) .+ abs.(mm)
    end
    storage .= abs2.(sqrt.(meas .+ b) .- sqrt.(μ .+ b))
    return sum(storage)
end


 # define custom gradient for speed-up
 # ChainRulesCore offers the possibility to define a backward AD rule
 # which can be used by several different AD systems
function ChainRulesCore.rrule(::typeof(anscombe_aux), μ, meas, storage; b=1)
    Y = anscombe_aux(μ, meas, storage, b=b)
    function anscombe_aux_pullback(xbar)
            storage .= xbar .* (one(eltype(μ)) .- sqrt.((meas .+ b) ./ (μ.+b)))
        return NoTangent(), storage, (ChainRulesCore.@not_implemented "Save computation"), (ChainRulesCore.@not_implemented "Save computation"), (ChainRulesCore.@not_implemented "Save computation")
    end

    return Y, anscombe_aux_pullback
end



"""
    Anscombe()
Returns a function to calculate Poisson loss using the Anscombe transform
Check the help of `anscombe_aux`.
"""

function Anscombe(b=1)
    (μ, meas, storage=similar(μ)) -> anscombe_aux(μ, meas, storage, b=b)
end
