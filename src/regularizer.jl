using Zygote: @adjoint, gradient
using Statistics
using Distributed

export GR, GR2
export Tikhonov
export TV


function Tikhonov(; λ=0.05, ϵ=1e-5)
    function f!(F, G, rec)
        if G != nothing
            G .= λ .* ∇_∇spatial_square(rec)
        end

        if F != nothing
            return λ .* sum(∇spatial_square(rec))
        end
    end
    return f! 
end

function TV(; λ=0.05, ϵ=1e-5)
    function f!(F, G, rec)
        c = spatial_diff_sqrt(rec)
        if G != nothing
            G .= λ .* ∇_spatial_diff_sqrt(rec) 
        end

        if F != nothing
            return λ * c / length(rec)
        end
    end
    
    return f!
end


function ∇_spatial_diff_sqrt(rec, ϵ=1e-5)
    out = zeros(eltype(rec), size(rec))
    for j = 2:size(rec)[2] - 1
        for i = 2:size(rec)[1] - 1
            #= out[i, j] = (-(rec[i+1, j] - rec[i, j]) + =# 
            #=             -(rec[i, j+1] - rec[i, j]))/ sqrt((rec[i+1, j] - rec[i, j])^2 + =# 
            #=             (rec[i, j+1] - rec[i, j])^2 + ϵ) =# 

            out[i, j] = (-(rec[i+1, j] - rec[i, j]) + -(rec[i-1, j] - rec[i, j]) 
                        -(rec[i, j+1] - rec[i, j]) -(rec[i, j-1] - rec[i, j]))/
                        sqrt((rec[i+1, j] - rec[i, j])^2 +  
                        (rec[i, j+1] - rec[i, j])^2 +
                        (rec[i, j-1] - rec[i, j])^2 +
                        (rec[i-1, j] - rec[i, j])^2)
        end
    end
    return 1 ./ length(rec) .* out
end

function spatial_diff_sqrt(rec)
    res = zero(first(rec))
    for j = 2:size(rec)[2] - 1
        for i = 2:size(rec)[1] - 1
            #= res += sqrt((rec[i+1, j] - rec[i, j])^2 + =#  
            #=             (rec[i, j+1] - rec[i, j])^2) =#
            res += sqrt((rec[i+1, j] - rec[i, j])^2 +  
                        (rec[i, j+1] - rec[i, j])^2 +
                        (rec[i, j-1] - rec[i, j])^2 +
                        (rec[i-1, j] - rec[i, j])^2)
        end
    end
    return  res
end


function GR(; λ=0.05, ϵ=1e-5)
    function f!(F, G, rec)
        precomputed =  ∇spatial_square2(rec)
        if G != nothing
            G .= λ .* ∇_GR(rec, length(rec) .* precomputed ./ 4, ϵ=ϵ) 
        end

        if F != nothing
            return λ .* sum(precomputed ./ (rec .+ ϵ))
        end
    end
    return f!
end

function GR2(; λ=0.05, ϵ=1e-5)
    function f!(F, G, rec)
        precomputed =  ∇spatial_square2(rec)
        if G != nothing
            G .= λ .* ReverseDiff.gradient(rec -> sum(∇spatial_square2(rec)), rec)
        end

        if F != nothing
            return λ .* sum(precomputed) 
        end
    end
    return f!
end

function ∇spatial_square2(rec)
    out = zeros(size(rec))
    R = CartesianIndices(rec)
    c_first, c_last = first(R), last(R)
    uc = oneunit(c_first)
    for I in R
        n, s = 0, zero(eltype(out))
        for J in max(c_first, I - uc):min(c_last, I + uc)
            n += 1
            s += (rec[I] - rec[J])^2
        end
        out[I] = s / n
    end
    return out ./ 4 ./ length(rec)
end

function ∇spatial_square(rec)
    res = zeros(size(rec))
        for j = 2:size(rec)[2] - 1
    for i = 2:size(rec)[1] - 1
            res[i, j] = ((rec[i, j + 1] - rec[i, j - 1])^2 +
                        (rec[i + 1, j] - rec[i - 1, j])^2)
        end
    end
    return res ./ 4 ./ length(rec)
end

function ∇_∇spatial_square(rec)
    res = zeros(size(rec))
        for j = 3:size(rec)[2] - 2
    for i = 3:size(rec)[1] - 2
            res[i, j] = 2 .* ((rec[i, j] - rec[i - 2, j])
                              + (rec[i, j] - rec[i, j - 2])
                              - (rec[i + 2, j] - rec[i, j])
                              - (rec[i, j + 2] - rec[i, j])) 
        end
    end
    return res ./ length(rec)
end


function ∇_GR(rec, precomputed; ϵ=1e-5)
    res = zeros(size(rec))
        for j = 3:size(rec)[2] - 2
    for i = 3:size(rec)[1] - 2
            res[i, j] = ((  (rec[i, j] - rec[i-2, j]) / (rec[i-1, j] + ϵ)
                         + (rec[i, j] - rec[i, j-2]) / (rec[i, j-1] + ϵ)
                         - (rec[i+2, j] - rec[i, j]) / (rec[i+1, j] + ϵ)
                         - (rec[i, j+2] - rec[i, j]) / (rec[i, j+1] + ϵ))
                         - precomputed[i, j] / (rec[i, j] + ϵ)^2)
        end
    end
    return res ./ 2 / length(rec)
end
