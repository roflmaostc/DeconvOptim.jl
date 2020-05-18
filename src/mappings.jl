using Zygote: @adjoint, gradient

export Non_negative
export map_0_1
export Exp

function map_0_1()
    f(x) = 1 .- exp.(.- x.^2)
    ∇f(x) = 2 .* x .* exp.(.- x.^2)
    f_inv(x) = sqrt.(.- log.(1 .- min.(1, x)))
    return f, ∇f, f_inv
end

function Exp()
    return ((x -> exp.(x)), (x -> exp.(x)) , (x -> log.(x)))
end


function IDm()
    return identity, (x -> 1), identity
end

function Non_negative()
    return non_negative, ∇non_negative, non_negative_inv
end


function non_negative(img)
    return img .^2
    #return 1 ./ (1 .+ exp.(.- img))
    #return sqrt.(0.1 .+  img.^2 ./ 4) .+ img ./ 2
end


function ∇non_negative(img)
    return 2 .* img
    #return .- exp.(.- img) ./ (1 .+ exp.(.- img)).^2
    #return 0.5 .+ img ./ 4 ./ sqrt.(0.1 .+ img.^2 ./ 4)
end
 @adjoint non_negative(img) = (non_negative(img),
                               c ->  (c .* ∇non_negative(img), ))

function non_negative_inv(img)
    return sqrt.(img)
    #return .- log.(1 ./ img .- 1)
    #return (img .^2 .- 0.1) ./ img
end
