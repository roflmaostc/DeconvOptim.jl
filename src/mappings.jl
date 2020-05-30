using Zygote: @adjoint, gradient

export Non_negative
export Non_negative2
export Non_negative3
export Map_0_1
export Exp

function Map_0_1()
    f(x) = 1 .- exp.(.- x.^2)
    ∇f(x) = 2 .* x .* exp.(.- x.^2)
    f_inv(x) = sqrt.(.- log.(1 .- min.(1, x)))
    return f, ∇f, f_inv
end


function Exp()
    return (x -> exp.(x)), (x -> exp.(x)) , (x -> log.(x))
end


function Non_negative()
    return (x -> x .^ 2), (x -> 2 .* x),  (x -> sqrt.(x))
end

function Non_negative3()
    function ∇f(x)
        if x > 0
            return 1
        elseif x < 0 
            return -1
        else
            return 0
        end
    end
    return (x -> abs.(x)), (x -> ∇f.(x)),  (x -> abs.(x))
end


function Non_negative2()
    f1(x) = 1 / (1 - x)
    ∇f1(x) = 1 / (1 - x) ^ 2
    f1_inv(y) = 1 - 1 / y


    f2(x) = (x + 0.5) ^ 2 + 0.75
    ∇f2(x) = 2 * (x + 0.5)
    f2_inv(y) = sqrt(y - 0.75)  - 0.5
    
    f(x) = x < 0 ? f1(x) : f2(x) 
    ∇f(x) = x < 0 ? ∇f1(x) : ∇f2(x) 
    f_inv(y) = y < 1 ? f1_inv(y) : f2_inv(y)

    return (x -> f.(x)), (x -> ∇f.(x)), (y -> f_inv.(y))
end



parabola(x) = x.^2
@adjoint parabola(x) = (f(x), c -> (c .* 2 .* x, ))
