using Zygote: @adjoint, gradient
using Statistics
using Distributed

export GR, GR2
export Tikhonov, Tikhonov_vec
export TV, TV_vec
export Positivity
export laplace

function laplace(rec)
    a = (rec[1:end-2, 2:end-1] .- 2 .* rec[2:end - 1, 2:end - 1] 
            .+ rec[3:end, 2:end - 1]) 
    b = (rec[2:end-1, 1:end-2] .- 2 .* rec[2:end - 1, 2:end - 1] 
            .+ rec[2:end - 1, 3:end])

    return sum((a .+ b) .^ 2)
end

function laplace(rec) 
    @tullio res = (rec[i - 1, j] + rec[i + 1, j] +   
                   rec[i, j + 1] + rec[i, j - 1] - 4 * rec[i,j])^2
    return res
end

function forward_gradient(rec)
    @tullio res = rec[i + 1, j] - rec[i, j] 
    return res 
end

function backward_gradient(rec)
    @tullio res = rec[i, j] - rec[i - 1, j]  
    return res 
end


function central_gradient(rec)
    @tullio res = 0.5 * (rec[i + 1, j] - rec[i - 1, j])  
    return res 
end


function Positivity(; λ=0.05)
    function f!(F, G, rec)
        if G != nothing
            G .= λ .* min.(0, rec) .* 2 ./ length(rec)
        end

        if F != nothing
            return λ * sum(min.(0, rec) .^ 2) / length(rec)
        end
    end
    return f!
end

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


function Tikhonov_vec(; λ=0.05, mode="laplace")


    if mode == "laplace"
        Γ = laplace_tullio
    elseif mode == "gradient"
        Γ = laplace 
    end

    function f!(F, G, rec)
        if G != nothing
            G .= λ .* gradient(Γ, rec)[1] / length(rec)
        end

        if F != nothing
            return λ * Γ(rec) / length(rec)
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

function TV_vec(; λ=0.05, mode="central", ϵ=1e-5)
    function total_var_center(rec)
        a = @view(rec[1:end - 2, 2:end - 1]) - @view(rec[3:end, 2:end - 1])
        b = @view(rec[2:end - 1, 1:end - 2]) - @view(rec[2:end - 1, 3:end])
        #= a = rec[1:end - 2, 2:end - 1] - rec[3:end, 2:end - 1] =#
        #= b = rec[2:end - 1, 1:end - 2] - rec[2:end - 1, 3:end] =#
        #= a = @view(rec[1:end - 2, 2:end - 1]) - @view(rec[3:end, 2:end - 1]) =#
        #= b = @view(rec[2:end - 1, 1:end - 2]) - @view(rec[2:end - 1, 3:end]) =#
        return 0.5 .* sum(sqrt.(a .^ 2 + b .^ 2))
    end

    function total_var_forward(rec)
        @tullio a = sqrt((rec[i, j] - rec[i + 1, j])^2 + 
                         (rec[i, j] - rec[i, j + 1])^2)
        return a
    end

    function total_var_backward(rec)
        a = rec[2:end - 1, 2:end - 1] - rec[1:end - 2, 2:end - 1]
        b = rec[2:end - 1, 2:end - 1] - rec[2:end - 1, 1:end - 2]
        return sum(sqrt.(a .^ 2 + b .^ 2))
    end
    
    function total_var_forward_backward(rec)
        # center - next 
        a = rec[2:end - 1, 2:end - 1] - rec[3:end, 2:end - 1]
        b = rec[2:end - 1, 2:end - 1] - rec[2:end - 1, 3:end]
        # center - previous
        c = rec[2:end - 1, 2:end - 1] - rec[1:end - 2, 2:end - 1]
        d = rec[2:end - 1, 2:end - 1] - rec[2:end - 1, 1:end - 2]
        return 0.5 .* sum(sqrt.(a .^ 2 + b .^ 2 + c .^2 + d .^ 2))
    end

    if mode == "central"
        total_var = total_var_center
    elseif mode == "forward_backward" 
        total_var = total_var_forward_backward
    elseif mode == "forward"
        total_var = total_var_forward
    elseif mode == "backward"
        total_var = total_var_backward
    end
    function f!(F, G, rec)
        c = spatial_diff_sqrt(rec)
        if G != nothing
            G .= λ .* gradient(total_var, rec)[1] / length(rec) 
        end

        if F != nothing
            return λ * total_var(rec) / length(rec)
        end
    end
    
    return f!
end

"""
    ∇_spatial_diff_sqrt(rec; ϵ=1e-5)

    This function calculates the average of the forward 
    and the backward spatial derivative.
"""
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
        precomputed =  ∇spatial_square(rec)
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
        precomputed =  ∇spatial_square(rec) ./ (rec .+ ϵ)
        if G != nothing
            G .= λ .* ReverseDiff.gradient(rec -> sum(∇spatial_square(rec)./ (rec .+ ϵ)), rec)[1]
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
