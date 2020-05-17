
using Zygote: @adjoint, gradient
using Statistics


export GR, TV



function ID(; λ=0.05)
    return (x -> 0), (x -> 0)
end

function TV(; λ=0.05)
    return (x -> λ .* TVf(x)), (x -> λ .* gradient(TVf, x)[1])
end

function TVf(img)
    dy = img[1:end - 2, 1:end - 2] .- img[3:end, 1:end - 2]
    dy_s = dy.^2
    dx = img[1:end - 2, 1:end - 2] .- img[1:end - 2, 3:end]
    dx_s = dx.^2
    return mean(dy_s .+ dx_s)
end


function GR(; λ=0.05)
    return (x -> λ .* GRf(x)), (x -> λ .* gradient(GRf, x)[1])
end

function GRf(img, ϵ=1e-5)
    ∇y = img[1:end - 2, 1:end - 2] .- img[3:end, 1:end - 2]
    ∇y_s = ∇y.^2
    ∇x = img[1:end - 2, 1:end - 2] .- img[1:end - 2, 3:end]
    ∇x_s = ∇x.^2
    return 0.25 * mean((∇x_s .+ ∇y_s) ./(img[2:end - 1, 2:end - 1] .+ ϵ))
end
