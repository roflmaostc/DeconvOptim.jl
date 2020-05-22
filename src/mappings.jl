using Zygote: @adjoint, gradient

export Non_negative
export Map_0_1
export Exp

function Map_0_1()
    f(x) = 1 .- exp.(.- x.^2)
    ∇f(x) = 2 .* x .* exp.(.- x.^2)
    f_inv(x) = sqrt.(.- log.(1 .- min.(1, x)))
    return f, ∇f, f_inv
end


function Exp()
    return ((x -> exp.(x)), (x -> exp.(x)) , (x -> log.(x)))
end


function Non_negative()
    return (x -> x .^ 2), (x -> 2 .* x),  (x -> sqrt.(x))
end

parabola(x) = x.^2
@adjoint parabola(x) = (f(x), c -> (c .* 2 .* x, ))
