
using Zygote: @adjoint, gradient
using Statistics


export GR, GR2, GR3, TV, TV2
export ∇s_square, ∇s_square2, ∇_∇s_square2


function ID(; λ=0.05)
    return (x -> 0), (x -> 0)
end


function TV2(; λ=0.05)
    return (x -> λ .* ∇s_square2(x), (x -> λ .* ∇_∇s_square2(x)))
end

function TV(; λ=0.05)
    return (x -> λ .* TVf(x)), (x -> λ .* gradient(TVf, x)[1])
end

function TVf(img)
    return mean(∇s_square(img))
end

function GR(; λ=0.05)
    function f!(F, G, rec)

        if G != nothing
            G = λ .* ∇_∇s_square2(rec)
        end

        if F != nothing
            return λ .* ∇s_square2(rec)
        end
    end
    return f!
end

function GR2(; λ=0.05, ϵ=1e-5)
    return (x -> λ .* GRf2(x)), (img -> 0.25 .* λ .* ∇_∇s_square2(img))
end

function GR3(; λ=0.05, ϵ=1e-5)
    return (x -> λ .* GRf2(x)), (img -> 0.25 .* λ .* gradient(∇s_square2, img)[1])
end

function GRf(img, ϵ=1e-5)
    return 0.25 ./ length(img[2:end-1, 2:end-1]) .* sum(∇s_square(img) ./ (img[2:end-1, 2:end-1] .+ ϵ))
end


function GRf2(img, ϵ=1e-5)
    return 0.25 .* ∇s_square2(img)
end

function ∇s_square(img)
    ∇y = img[3:end, 2:end - 1] .- img[1:end - 2, 2:end - 1]
    ∇x = img[2:end - 1, 3:end] .- img[2:end - 1, 1:end - 2]
    return (∇x .^2 .+ ∇y .^2) ./ 4
end



function ∇s_square2(img; ϵ=1e-5)
    res = 0
    for i = 2:size(img)[1] - 1
        for j = 2:size(img)[2] - 1
            res = res + ((img[i, j + 1] - img[i, j - 1])^2 +
                (img[i + 1, j] - img[i - 1, j])^2) / (img[i,j] + ϵ)
        end
    end
    return res / 4 / length(img[2:end-1, 2:end-1])
end
@adjoint ∇s_square2(img) = ∇s_square2(img), c -> (c * ∇_∇s_square2(img),)

function ∇_∇s_square2(img, ϵ=1e-5)
    res = zeros(size(img))
    for i = 3:size(img)[1] - 2
        for j = 3:size(img)[2] - 2
            res[i, j] = 2 *   (1*(img[i, j]   - img[i-2, j])/(img[i-1, j]  + ϵ)
                              +1* (img[i, j]   - img[i, j-2])/(img[i,   j-1]+ ϵ)
                              -1* (img[i+2, j] - img[i,j]   )/(img[i+1, j]  + ϵ)
                              -1*  (img[i, j+2] - img[i,j]   )/(img[i,   j+1]+ ϵ))-
                            ((img[i, j + 1] - img[i, j - 1])^2 + (img[i + 1, j] - img[i - 1, j])^2) / (img[i,j] + ϵ)^2
        end
    end
    return res ./ 4 / length(img[2:end - 1, 2:end - 1])
end
