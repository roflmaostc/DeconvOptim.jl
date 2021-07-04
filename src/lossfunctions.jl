export Poisson, poisson_aux
export Gauss, gauss_aux
export ScaledGauss, scaled_gauss_aux
export Anscombe, anscombe_aux
using Tullio

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
function scaled_gauss_aux(μ, meas, storage=copy(μ); read_var=1)
    # μ[μ .<= 1f-8] .= 1f-8
    storage .= log.(μ .+read_var) .+ (meas .- μ).^2 ./ (μ.+read_var)
    return sum(storage)
end

 # define custom gradient for speed-up
function ChainRulesCore.rrule(::typeof(scaled_gauss_aux), μ, meas, storage=copy(μ); read_var=1)
    Y = scaled_gauss_aux(μ, meas, storage, read_var= read_var) 
    function scaled_gauss_aux_pullback(xbar)
        ∇ = (μ.^2 .- meas.^2 .+ μ .+ read_var.*(1 .- 2 .* (meas .- µ)))./((μ.+read_var).^2)
        # ∇[μ .<= 1f-8] .= 0 
        return zero(eltype(μ)), ∇, zero(eltype(μ)), zero(eltype(μ)), zero(eltype(μ)) 
    end
    return Y, scaled_gauss_aux_pullback
end


"""
    ScaledGauss()

Returns a function to calculate scaled Gauss loss.
Check the help of `scaled_gauss_aux`.
"""
function ScaledGauss(read_var=1)
    return (µ, meas, storage=copy(µ)) -> scaled_gauss_aux(µ, meas, storage, read_var=read_var)
end

"""
    anscombe_aux(μ, meas, storage=copy(μ))

Calculates the Poisson loss using the Anscombe-based norm for `μ` and `meas`.
`μ` can be of larger size than `meas`. In that case
we extract a centered region from `μ` of the same size as `meas`.
""" 
function anscombe_aux(μ, meas, storage=copy(μ); b=1, sqrt_meas_b=0)
    # due to numerical errors, μ can be negative or 0
    mm = minimum(μ)
    if mm <= 0
        μ .= μ .+ eps(maximum(μ)) .+ abs.(mm)
    end
    storage .= abs2.(sqrt_meas_b .- sqrt.(μ .+ b))
    return sum(storage)
    #@tullio s = @inbounds abs2(sqrt_meas_b[i] - sqrt(μ[i] + b))
    #return s
end


 # define custom gradient for speed-up
 # ChainRulesCore offers the possibility to define a backward AD rule
 # which can be used by several different AD systems
function ChainRulesCore.rrule(::typeof(anscombe_aux), μ, meas, storage=copy(μ); b=1, sqrt_meas_b=0)
    Y = anscombe_aux(μ, meas, storage; b=b, sqrt_meas_b=sqrt_meas_b)
    function anscombe_aux_pullback(xbar)
        # @show xbar
        # @tullio storage[i] = @inbounds (one(eltype(μ)) - sqrt_meas_b[i] / sqrt(μ[i] + b)) # xbar[i]
        storage .= (one(eltype(μ)) .- sqrt_meas_b ./ sqrt.(μ.+b)) #xbar .* 
        return zero(eltype(μ)), storage, zero(eltype(μ)), zero(eltype(storage)), zero(eltype(b)), zero(eltype(μ))
    end

    return Y, anscombe_aux_pullback
end


"""
    Anscombe()

Returns a function to calculate Poisson loss using the Anscombe transform
Check the help of `anscombe_aux`.
"""

function Anscombe(b=1)
    CallCounter = 0;
    sqrt_meas_b = 0;
    function call_brancher(μ, meas, storage=copy(µ)) # uses a closure to deal with possibly normalized measurement data.
        if CallCounter == 0
            sqrt_meas_b = sqrt.(meas .+ b);
            CallCounter += 1;
        end
        return anscombe_aux(μ, meas, storage; b=b, sqrt_meas_b=sqrt_meas_b)
    end
    return call_brancher;
end

