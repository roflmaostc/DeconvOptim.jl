using Zygote
using Tullio

export GR, GR_old
export Tikhonov_old, Tikhonov
export TV, TV_old
export Positivity
export laplace
export generate_spatial_grad_square_n

 # First we define several functions which are helpful
 # to use in some of the regularizers
function laplace(rec) 
    # todo proper handling for 3D and 2D because laplacian
    # is defined differently
    @tullio res = (rec[i - 1, j, k, l, m] + rec[i + 1, j, k, l, m] +   
                   rec[i, j - 1, k, l, m] + rec[i, j + 1, k, l, m] +   
                   rec[i, j, k - 1, l, m] + rec[i, j, k + 1, l, m] - 
                   4 * rec[i,j])^2
    return res
end


 # meta programing
 # Reference and thanks to @mcabbott
 # https://github.com/mcabbott/Tullio.jl/issues/11
 #
 # this function creates n dimensional Tullio functions for calculating
 # some regularizers
 # expr is the parameter which performs the operation of the regularizer
 # comb_f is a function combinining these elementary operations over
 #      dimensions
 # off1 and off2 are offsets of the filter
 #
function create_Ndim_regularizer(expr, comb_f, num_dim, sum_dim)
    out, add = [], []
    for d in 1:sum_dim
        ind = :i # @gensym ind
        inds1 = map(1:num_dim) do di
            i = Symbol(ind, di)
            di == d ? :($i+1) : i
        end
        inds2 = map(1:num_dim) do di
            i = Symbol(ind, di)
            di == d ? :($i-1) : i
        end
        push!(add, expr(inds1, inds2))
    end
    if comb_f == nothing
        push!(out, :(@tullio res = +($(add...))))
    else
        push!(out, :(@tullio res = $comb_f(+($(add...)))))
    end
    return out
end

 # function to generate a spatial_grad_square filter for n dimensions
function generate_spatial_grad_square_n(num_dim, sum_dim)
    expr(inds1, inds2) = :(abs2(arr[$(inds1...)] - arr[$(inds2...)]))
    comb_f = nothing#:abs2 
    @eval x = arr -> ($(create_Ndim_regularizer(expr, comb_f,
                                                 num_dim, sum_dim)...))
    return x
end


 # quadrat für rotationsinvarianz
 # manhatten norm
function forward_gradient(rec)
    res1 = let 
        @tullio res1 = abs2(rec[i + 1, j, k, l, m] - rec[i, j, k, l, m])
    end
    res2 = let 
        @tullio res2 = abs2(rec[i, j + 1, k, l, m] - rec[i, j, k, l, m]) 
    end
    res3 = let 
        @tullio res3 = abs2(rec[i, j, k + 1, l, m] - rec[i, j, k, l, m])
    end
    return res1 + res2 + res3
end


function central_gradient(rec)
    @tullio res = 0.5 * (abs2(rec[i - 1, j, k, l, m] - rec[i + 1, j, k, l, m]) +  
                         abs2(rec[i, j - 1, k, l, m] - rec[i, j + 1, k, l, m]) + 
                         abs2(rec[i, j, k - 1, l, m] - rec[i, j, k + 1, l, m])) 
    return res 
end


# Tikhonov is the preferred implementation over Tikhonov_old
function Tikhonov(; λ=0.05, mode="laplace")

    if mode == "laplace"
        Γ = laplace
    elseif mode == "spatial_grad_square"
        # this line needs to be fixed
        # Tikhonov must have num_dim and sum_dim as parameters to indicate
        # which axis must be sum and which axes are present
        Γ = generate_spatial_grad_square_n(5,2)#spatial_grad_square
    elseif mode == "forward_grad"
        Γ = forward_gradient 
    elseif mode == "central_grad"
        Γ = central_gradient 
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



 # √(abs2(rec) + ϵ)
 # x²  
function GR(; λ=0.05, mode="central", ϵ=1e-5)
    function GR_central(rec)
        @tullio reg = ((rec[i - 1, j, k, l, m] - rec[i + 1, j, k, l, m])^2 + 
                       (rec[i, j - 1, k, l, m] - rec[i, j + 1, k, l, m])^2 + 
                       (rec[i, j, k - 1, l, m] - rec[i, j, k + 1, l, m])^2) / 
                      (rec[i, j, k, l, m] + ϵ)
        return reg 
    end
    function GR_forward(rec)
        @tullio reg = ((rec[i, j, k, l, m] - rec[i + 1, j, k, l, m])^2 + 
                       (rec[i, j, k, l, m] - rec[i, j + 1, k, l, m])^2 + 
                       (rec[i, j, k, l, m] - rec[i, j, k + 1, l, m])^2) / 
                      (rec[i, j, k, l, m] + ϵ)
        return reg
    end

    if mode == "central"
        GRf = GR_central 
    elseif mode == "forward"
        GRf = GR_forward 
    end

    function f!(F, G, rec)
        if G != nothing
            G .= λ ./ 4 .* gradient(GRf, rec)[1] / length(rec) 
        end
        if F != nothing
            return λ * GRf(rec) / 4 / length(rec) 
        end
    end
    return f!
end



function TV(; λ=0.05, mode="central")
    function total_var_center(rec)
        @tullio reg = sqrt((rec[i - 1, j, k, l, m] - rec[i + 1, j, k, l, m])^2 + 
                           (rec[i, j - 1, k, l, m] - rec[i, j + 1, k, l, m])^2 +
                           (rec[i, j, k - 1, l, m] - rec[i, j, k + 1, l, m])^2)
        return reg 
    end

    function total_var_forward(rec)
        @tullio reg = sqrt((rec[i, j, k, l, m] - rec[i + 1, j, k, l, m])^2 + 
                           (rec[i, j, k, l, m] - rec[i, j + 1, k, l, m])^2 +
                           (rec[i, j, k, l, m] - rec[i, j, k + 1, l, m])^2)
        return reg
    end

    if mode == "central"
        total_var = total_var_center
    elseif mode == "forward"
        total_var = total_var_forward
    end

    # definition of the function which will be called by Optim
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









# Old stuff from here on which will be removed

function Tikhonov_old(; λ=0.05, ϵ=1e-5)
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


function TV_old(; λ=0.05, ϵ=1e-5)
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


function GR_old(; λ=0.05, ϵ=1e-5)
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
