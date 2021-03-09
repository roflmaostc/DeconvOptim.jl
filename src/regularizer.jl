using Zygote
using Tullio

export Tikhonov, GR, TV
export generate_spatial_grad_square, generate_GR, generate_TV


 # General hint
 # for the creation of the regularizers we are using meta programming
 # because the fastest way for automatic differentiation and Zygote
 # is Tullio.jl at the moment. 
 # Our metaprogramming code was initially based on
 # https://github.com/mcabbott/Tullio.jl/issues/11
 # 


"""
    generate_indices(num_dims, d, ind1, ind2)
Generates a list of symbols which can be used to generate Tullio expressions
via metaprogramming.
`num_dims` is the total number of dimensions.
`d` is the dimension where there is a offset in the index.
`ind1` and `ind2` are the offsets each.
 # Examples
 ```julia-repl
julia> a, b = generate_indices(5, 2, 1, 1)
(Any[:i1, :(i2 + 1), :i3, :i4, :i5], Any[:i1, :(i2 + 1), :i3, :i4, :i5])
```
"""
function generate_indices(num_dims, d, ind1, ind2)
    # create initial symbol
    ind = :i
    # create the array of symbols for each dimension
    inds1 = map(1:num_dims) do di
        # map over numbers and append the number of the position to symbol i
        i = Symbol(ind, di)
        # at the dimension where we want to do the step, add $ind1
        di == d ? :($i + $ind1) : i
    end
    inds2 = map(1:num_dims) do di
        i = Symbol(ind, di)
        # here we do the step but in the other dimension. ind2 should be 
        # (-1) * ind1 or negative
        di == d ? :($i + $ind2) : i
    end
    return inds1, inds2
end


"""
    generate_laplace(num_dims, sum_dims_arr, weights)
Generate the Tullio statement for computing the abs2 of Laplacian.
`num_dims` is the dimension of the array. 
`sum_dims_arr` is a array indicating over which dimensions we must sum over.
`weights` is a array of a weight for the different dimension.
"""
function generate_laplace(num_dims, sum_dims_arr, weights; debug=false)
    # create out list for the final expression
    # add accumulates the different add expressions
    out, add = [], []
    # loop over all dimensions which we want to sum. for each dimension must
    # be a weight provided
    for (d, w) in zip(sum_dims_arr, weights)
        # get the two lists of indices
        inds1, inds2 = generate_indices(num_dims, d, 1, -1) 
        # critical part where we actually add the two expressions for
        # the steps in the dimension to the add array
        push!(add, :($w * arr[$(inds1...)] + $w * arr[$(inds2...)]))
    end
    # for laplace we need one final negative term at the position itself
    inds = map(1:num_dims) do di
        i = Symbol(:i, di)
    end
    # subtract this final term
    pre_factor = 2 ^ num_dims * sum(weights)
    push!(add, :(-$:($pre_factor * arr[$(inds...)])))
    # create final expressions by adding all elements of the add list
    if debug
        push!(out, :(res = abs2(+$(add...))))
    else
        push!(out, :(@tullio  res = abs2(+$(add...))))
    end
    return out
end


"""
    create_Ndim_regularizer(expr, num_dims, sum_dims_arr, weights, ind1, ind2)
    A helper function to create a N-dimensional regularizer. In principle
    the same as `generate_laplace` but more general
    `expr` needs to be a function which takes `inds1`, `inds2` and a weight `w`-
    `num_dims` is the total amount of dimensions
    `sum_dims_arr` is a array indicating over which dimensions we must sum over.
    `weights` is a array of a weight for the different dimension.
    ``
"""
function create_Ndim_regularizer(expr, num_dims, sum_dims_arr, weights, 
                                 ind1, ind2)
    out, add = [], []
    for (d, w) in zip(sum_dims_arr, weights)
        inds1, inds2 = generate_indices(num_dims, d, ind1, ind2) 
        push!(add, expr(inds1, inds2, w))
    end
    push!(out, :(@tullio  res = +($(add...))))
    return out
end


""" 
    generate_spatial_grad_square(num_dims, sum_dims_arr, weights)
Generate the Tullio statement for calculating the squared spatial gradient
over n dimensions.
`num_dims` is the dimension of the array. 
`sum_dims_arr` is a array indicating over which dimensions we must sum over.
`weights` is a array of a weight for the different dimension.
`ind1` and `ind2` are the offsets for the difference.
"""
function generate_spatial_grad_square(num_dims, sum_dims_arr, weights)
    expr(inds1, inds2, w) = :($w * abs2(arr[$(inds1...)] - arr[$(inds2...)]))
    @eval x = arr -> ($(create_Ndim_regularizer(expr, num_dims, sum_dims_arr, 
                        weights, 1, -1)...))
    return x
end


"""
    Tikhonov(; <keyword arguments>)
This function returns a function to calculate the Tikhonov regularizer
of a n-dimensional array. 
# Arguments
- `num_dims=2`: 
- `sum_dims=[1, 2]`: A array containing the dimensions we want to sum over
- `weights=nothing`: A array containing weights to weight the contribution of 
    different dimensions. If `weights=nothing` all dimensions are weighted equally.
- `step=1`: A integer indicating the step width for the array indexing
- `mode="laplace"`: Either `"laplace"`, `"spatial_grad_square"`, `"identity"` accounting for different
    modes of the Tikhonov regularizer. Default is `"laplace"`.
# Examples
To create a regularizer for a 3D dataset where the third dimension
has different contribution.
```julia-repl
julia> reg = Tikhonov(num_dims=2, sum_dims=[1, 2], weights=[1, 1], mode="identity");

julia> reg([1 2 3; 4 5 6; 7 8 9])
285
```
"""
function Tikhonov(;num_dims=2, sum_dims=[1, 2], weights=[1, 1], step=1, mode="laplace")
    if weights == nothing
        weights = ones(Int, num_dims)
    end
    if mode == "laplace"
        Γ = @eval arr -> ($(generate_laplace(num_dims, sum_dims, weights)...))
    elseif mode == "spatial_grad_square"
        expr(inds1, inds2, w) = :($w * abs2(arr[$(inds1...)] - arr[$(inds2...)]))
        Γ = @eval arr -> ($(create_Ndim_regularizer(expr, num_dims, sum_dims, 
                            weights, step, (-1) * step)...))
    elseif mode == "identity"
        Γ = arr -> sum(abs2.(arr))
    else
        throw(ArgumentError("The provided mode is not valid."))
    end

    return Γ
end



"""
    generate_GR(num_dims, sum_dims_arr, weights, ind1, ind2, ϵ)
Generate the Tullio statement for computing the Good's roughness.
`num_dims` is the dimension of the array. `sum_dims_arr` is a array
indicating over which dimensions we must sum over.
`weights` is a array of a weight for the different dimension.
`ind1` and `ind2` are the offsets for the difference.
"""
function generate_GR(num_dims, sum_dims_arr, weights, ind1, ind2; debug=false)
    out, add = [], []
    inds = map(1:num_dims) do di
        i = Symbol(:i, di)
    end
    
    for (d, w) in zip(sum_dims_arr, weights)
        inds1, inds2 = generate_indices(num_dims, d, ind1, ind2) 
        push!(add, :($w * (arr[$(inds1...)] + arr[$(inds2...)])))
    end
    prefactor = - 4 / (abs(ind1) + abs(ind2))
    diff_factor = -sum(weights) * 2
    push!(add, :($diff_factor *arr[$(inds...)]))
    if debug
        push!(out, :(res = $prefactor * arr[$(inds...)] * +($(add...))))
    else
        push!(out, :(@tullio  res = $prefactor * arr[$(inds...)] * +($(add...))))
    end
    return out
end


"""
    GR(; <keyword arguments>)
This function returns a function to calculate the Good's roughness regularizer
of a n-dimensional array. 
# Arguments
- `num_dims=2`: Dimension of the array that should be regularized 
- `sum_dims=[1, 2]`: A array containing the dimensions we want to sum over
- `weights=nothing`: A array containing weights to weight the contribution of 
    different dimensions. If `weights=nothing` all dimensions are weighted equally.
- `step=1`: A integer indicating the step width for the array indexing
- `mode="forward"`: Either `"central"` or `"forward"` accounting for different
    modes of the spatial gradient. Default is "central".
# Examples
To create a regularizer for a 3D dataset where the third dimension
has different contribution. For the derivative we use forward mode.
```julia-repl
julia> reg = GR(num_dims=2, sum_dims=[1, 2], weights=[1, 1], mode="forward");

julia> reg([1 2 3; 4 5 6; 7 8 9])
-26.36561871738898
```
"""
function GR(; num_dims=2, sum_dims=[1, 2], weights=[1, 1], step=1,
              mode="central", ϵ=1f-8)
    if weights == nothing
        weights = ones(Int, num_dims)
    end
    if mode == "central"
        GRf = @eval arr -> ($(generate_GR(num_dims, sum_dims, weights,
                                        step, (-1) * step)...))
    elseif mode == "forward"
        GRf = @eval arr -> ($(generate_GR(num_dims, sum_dims, weights,
                                        step, 0)...))
    else
        throw(ArgumentError("The provided mode is not valid."))
    end
    
    # we need to add a ϵ to prevent NaN in the derivative of it
    return arr -> GRf(sqrt.(arr .+ ϵ))
end


"""
    generate_TV(num_dims, sum_dims_arr, weights, ind1, ind2, ϵ)
Generate the Tullio statement for computing the Good's roughness.
`num_dims` is the dimension of the array. 
`sum_dims_arr` is a array
indicating over which dimensions we must sum over.
`weights` is a array of a weight for the different dimension.
`ind1` and `ind2` are the offsets for the difference.
`ϵ` is a numerical constant to prevent division by zero. 
    this is important for the gradient 
"""
function generate_TV(num_dims, sum_dims_arr, weights, ind1, ind2, ϵ=1f-8; debug=false)
    out, add = [], []
    for (d, w) in zip(sum_dims_arr, weights)
        inds1, inds2 = generate_indices(num_dims, d, ind1, ind2) 
        push!(add, :($w * abs2(arr[$(inds1...)] - arr[$(inds2...)])))
    end
    push!(add, ϵ)
    if debug
        push!(out, :(res = sqrt(+($(add...)))))
    else
        push!(out, :(@tullio  res = sqrt(+($(add...)))))
    end
    return out
end


"""
    TV(; <keyword arguments>)
This function returns a function to calculate the Total Variation regularizer
of a n-dimensional array. 
# Arguments
- `num_dims=2`: 
- `sum_dims=[1, 2]`: A array containing the dimensions we want to sum over
- `weights=nothing`: A array containing weights to weight the contribution of 
    different dimensions. If `weights=nothing` all dimensions are weighted equally.
- `step=1`: A integer indicating the step width for the array indexing
- `mode="central"`: Either `"central"` or `"forward"` accounting for different
    modes of the spatial gradient. Default is "central".
# Examples
To create a regularizer for a 3D dataset where the third dimension
has different contribution. For the derivative we use forward mode.
```julia-repl
julia> reg = TV(num_dims=2, sum_dims=[1, 2], weights=[1, 1], mode="forward");

julia> reg([1 2 3; 4 5 6; 7 8 9])
12.649111f0
```
"""
function TV(; num_dims=2, sum_dims=[1, 2], weights=nothing, step=1, mode="central")
    
    if weights == nothing
        weights = ones(Int, num_dims)
    end

    if mode == "central"
        total_var = @eval arr -> ($(generate_TV(num_dims, sum_dims, weights,
                                        step, (-1) * step)...))
    elseif mode == "forward"
        total_var = @eval arr -> ($(generate_TV(num_dims, sum_dims, weights,
                                        step, 0)...))
    else
        throw(ArgumentError("The provided mode is not valid."))
    end
    return total_var
end

