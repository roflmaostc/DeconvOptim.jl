export Non_negative
export Map_0_1
export Piecewise_positive
export Pow4_positive
export Abs_positive
 

 # All these functions return a mapping function and 
 # the inverse of it 
 # they are used to map the real numbers to non-negative real numbers


"""
    Non_negative()

Returns a function and an inverse function inverse function
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

Returns a function and an inverse function
to map numbers to an interval between 0 and 1.
via an exponential function.

"""
function Map_0_1()
    return f01, f01_inv
end

f01(x) = 1 .- exp.(.- x.^2)
f01_inv(y) = sqrt.(.- log.(1 .- y))

"""
    Piecewise_positive()

Returns a function and an inverse function 
to map numbers to larger than 0 via
two function stitched together.
"""
function Piecewise_positive()
    return f_pw_pos, f_pw_pos_inv
end

f_pw_pos(x) = ifelse.(x .> 0, one(eltype(x)) .+ x,one(eltype(x))./(one(eltype(x)).-x))
f_pw_pos_grad(x) = ifelse.(x .> 0, one(eltype(x)) , one(eltype(x))./abs2.(one(eltype(x)).-x))
f_pw_pos_inv(y) = ifelse.(y .> 1, y .- one(eltype(y)), one(eltype(y)) .- one(eltype(y))./y)

function ChainRulesCore.rrule(::typeof(f_pw_pos), x)
    Y = f_pw_pos(x)
    function aux_pullback(barx)
        return zero(eltype(Y)), barx .* f_pw_pos_grad(x)
    end
    return Y, aux_pullback
end


"""
    Pow4_positive()

Returns a function and an inverse function
to map numbers to larger than 0 with `abs2.(abs2.(x))`

"""
function Pow4_positive()
    return f_pow4, f_pow4_inv
end

f_pow4(x) = abs2.(abs2.(x))
f_pow4_inv(y) = sqrt.(sqrt.(y))


"""
    Abs_positive()

Returns a function and an inverse function
to map numbers to larger than 0. 
"""
function Abs_positive()
    return f_abs, f_abs_inv
end

f_abs(x) = abs.(x)
f_abs_inv(y) = abs.(y)
