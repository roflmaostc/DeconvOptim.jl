export Non_negative
export Map_0_1
 

 # All these functions return a mapping function and 
 # the inverse of it 
 # they are used to map the real numbers to non-negative real numbers


"""
    Non_negative()

Returns a function and a inverse function inverse function
to map numbers to non-negative numbers.
We use a parabola.

# Examples
```julia-repl
julia> p, p_inv = Non_negative()
(DeconvOptim.var"#5#7"(), DeconvOptim.var"#6#8"())

julia> x = [-1, 2, -3]
3-element Array{Int64,1}:
 -1
  2
 -3

julia> p(x)
3-element Array{Int64,1}:
 1
 4
 9

julia> p_inv(p(x))
3-element Array{Float64,1}:
 1.0
 2.0
 3.0
```
"""
function Non_negative()
    #return x -> map(abs2, x), parab_inv 
    return parab, parab_inv 
end

parab(x) = abs2.(x)
parab_inv(x) = sqrt.(x)
 # define custom adjoint for parab because of 
 # slow broadcasting
function ChainRulesCore.rrule(::typeof(parab), x)
    Y = parab(x)
    function aux_pullback(barx)
        return zero(eltype(Y)), (2 .* barx) .* x
    end
    return Y, aux_pullback
end


"""
    Map_0_1()

Returns a function and a inverse function inverse function
to map numbers to an interval between 0 and 1. 

"""
function Map_0_1()
    return f01, f01_inv
end

f01(x) = 1 .- exp.(.- x.^2)
f01_inv(y) = sqrt.(.- log.(1 .- y))
