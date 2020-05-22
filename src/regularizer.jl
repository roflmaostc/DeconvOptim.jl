
using Zygote: @adjoint, gradient
using Statistics


export GR
export TV
export ∇_∇spatial_square



function TV(; λ=0.05)
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


function GR(; λ=0.05, ϵ=1e-5)
    function f!(F, G, rec)
        precomputed =  ∇spatial_square(rec)
        if G != nothing
            G .= λ .* ∇_GR(rec, length(rec) .* precomputed ./ 4) 
        end

        if F != nothing
            return λ .* sum(precomputed) 
        end
    end
    return f!
end



function ∇spatial_square(rec; ϵ=1e-5)
    res = zeros(size(rec))
    for i = 2:size(rec)[1] - 1
        for j = 2:size(rec)[2] - 1
            res[i, j] = ((rec[i, j + 1] - rec[i, j - 1])^2 +
                        (rec[i + 1, j] - rec[i - 1, j])^2)
        end
    end
    return res ./ 4 ./ length(rec)
end

function ∇_∇spatial_square(rec; ϵ=1e-5)
    res = zeros(size(rec))
    for i = 3:size(rec)[1] - 2
        for j = 3:size(rec)[2] - 2
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
    for i = 3:size(rec)[1] - 2
        for j = 3:size(rec)[2] - 2
            res[i, j] = ((  (rec[i, j] - rec[i-2, j]) / (rec[i-1, j] + ϵ)
                         + (rec[i, j] - rec[i, j-2]) / (rec[i, j-1] + ϵ)
                         - (rec[i+2, j] - rec[i, j]) / (rec[i+1, j] + ϵ)
                         - (rec[i, j+2] - rec[i, j]) / (rec[i, j+1] + ϵ))
                         - precomputed[i, j] / (rec[i, j] + ϵ)^2)
        end
    end
    return res ./ 2 / length(rec)
end
