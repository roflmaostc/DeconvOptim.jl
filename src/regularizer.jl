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
    generate_indices(num_dim, d, ind1, ind2)

Generates a list of symbols which can be used to generate Tullio expressions
via metaprogramming.
`num_dim` is the total number of dimensions.
`d` is the dimension where there is a offset in the index.
`ind1` and `ind2` are the offsets each.

 # Examples
 ```julia-repl
julia> a, b = generate_indices(5, 2, 1, 1)
(Any[:i1, :(i2 + 1), :i3, :i4, :i5], Any[:i1, :(i2 + 1), :i3, :i4, :i5])
```
"""
function generate_indices(num_dim, d, ind1, ind2)
    # create initial symbol
    ind = :i
    # create the array of symbols for each dimension
    inds1 = map(1:num_dim) do di
        # map over numbers and append the number of the position to symbol i
        i = Symbol(ind, di)
        # at the dimension where we want to do the step, add $ind1
        di == d ? :($i + $ind1) : i
    end
    inds2 = map(1:num_dim) do di
        i = Symbol(ind, di)
        # here we do the step but in the other dimension. ind2 should be 
        # (-1) * ind1 or negative
        di == d ? :($i + $ind2) : i
    end
    return inds1, inds2
end


"""
    generate_laplace(num_dim, sum_dims_arr, weights)

Generate the Tullio statement for computing the Laplacian.
`num_dim` is the dimension of the array. 
`sum_dims_arr` is a array indicating over which dimensions we must sum over.
`weights` is a array of a weight for the different dimension.
"""
function generate_laplace(num_dim, sum_dims_arr, weights)
    # create out list for the final expression
    # add accumulates the different add expressions
    out, add = [], []
    # loop over all dimensions which we want to sum. for each dimension must
    # be a weight provided
    for (d, w) in zip(sum_dims_arr, weights)
        # get the two lists of indices
        inds1, inds2 = generate_indices(num_dim, d, 1, -1) 
        # critical part where we actually add the two expressions for
        # the steps in the dimension to the add array
        push!(add, :($w * arr[$(inds1...)] + $w * arr[$(inds2...)]))
    end
    # for laplace we need one final negative term at the position itself
    inds = map(1:num_dim) do di
        i = Symbol(:i, di)
    end
    # subtract this final term
    push!(add, :(-$(2 * length(sum_dims_arr)) * arr[$(inds...)]))
    # create final expressions by adding all elements of the add list
    push!(out, :(@tullio res = abs2(+$(add...))))
    return out
end


"""
    create_Ndim_regularizer(expr, num_dim, sum_dims_arr, weights, ind1, ind2)

    A helper function to create a N-dimensional regularizer. In principle
    the same as `generate_laplace` but more general

    `expr` needs to be a function which takes `inds1`, `inds2` and a weight `w`-
    `num_dim` is the total amount of dimensions
    `sum_dims_arr` is a array indicating over which dimensions we must sum over.
    `weights` is a array of a weight for the different dimension.
    ``
"""
function create_Ndim_regularizer(expr, num_dim, sum_dims_arr, weights, 
                                 ind1, ind2)
    out, add = [], []
    for (d, w) in zip(sum_dims_arr, weights)
        inds1, inds2 = generate_indices(num_dim, d, ind1, ind2) 
        push!(add, expr(inds1, inds2, w))
    end
    push!(out, :(@tullio res = +($(add...))))
    return out
end


""" 
    generate_spatial_grad_square(num_dim, sum_dims_arr, weights)

Generate the Tullio statement for calculating the squared spatial gradient
over n dimensions.
`num_dim` is the dimension of the array. 
`sum_dims_arr` is a array indicating over which dimensions we must sum over.
`weights` is a array of a weight for the different dimension.
`ind1` and `ind2` are the offsets for the difference.
"""
function generate_spatial_grad_square(num_dim, sum_dims_arr, weights)
    expr(inds1, inds2, w) = :($w * abs2(arr[$(inds1...)] - arr[$(inds2...)]))
    @eval x = arr -> ($(create_Ndim_regularizer(expr, num_dim, sum_dims_arr, 
                        weights, 1, -1)...))
    return x
end


"""
    Tikhonov(; <keyword arguments>)

This function returns a function to calculate the Tikhonov regularizer
of a n-dimensional array. 

# Arguments
- `num_dim=5`: 
- `sum_dims=[1, 2]`: A array containing the dimensions we want to sum over
- `weights=[1, 1]`: A array containing weights to weight the contribution of 
    different dimensions
- `step=1`: A integer indicating the step width for the array indexing
- `mode="laplace"`: Either `"laplace"` or `"spatial_grad_square"` accounting for different
    modes of the Tikhonov regularizer. Default is `"laplace"`.

# Examples
To create a regularizer for a 3D dataset where the third dimension
has a different sampling (factor 4 larger) than the first two dimensions.

```julia-repl
julia> Tikhonov(sum_dims=[1, 2, 3], weights=[1, 1, 0.25])
```
"""
function Tikhonov(;num_dim=5, sum_dims=[1, 2], weights=[1, 1], step=1, mode="laplace")
    if mode == "laplace"
        Γ = @eval arr -> ($(generate_laplace(num_dim, sum_dims, weights)...))
    elseif mode == "spatial_grad_square"
        expr(inds1, inds2, w) = :($w * abs2(arr[$(inds1...)] - arr[$(inds2...)]))
        Γ = @eval arr -> ($(create_Ndim_regularizer(expr, num_dim, sum_dims, 
                            weights, step, (-1) * step)...))
    end

    return rec -> Γ(rec) / length(rec)
end



"""
    generate_GR(num_dim, sum_dims_arr, weights, ind1, ind2, ϵ)

Generate the Tullio statement for computing the Good's roughness.
`num_dim` is the dimension of the array. `sum_dims_arr` is a array
indicating over which dimensions we must sum over.
`weights` is a array of a weight for the different dimension.
`ind1` and `ind2` are the offsets for the difference.
`ϵ` is a numerical constant to prevent division by zero.
"""
function generate_GR(num_dim, sum_dims_arr, weights, ind1, ind2, ϵ)
    out, add = [], []
    for (d, w) in zip(sum_dims_arr, weights)
        inds1, inds2 = generate_indices(num_dim, d, ind1, ind2) 
        push!(add, :(abs2($w * arr[$(inds1...)] - $w * arr[$(inds2...)])))
    end
    inds = map(1:num_dim) do di
        i = Symbol(:i, di)
    end
    denom = :(sqrt(abs2(arr[$(inds...)])) + $ϵ)
    push!(out, :(@tullio res = +($(add...)) / $denom))
    return out
end


"""
    GR(; <keyword arguments>)

This function returns a function to calculate the Good's roughness regularizer
of a n-dimensional array. 

# Arguments
- `num_dim=5`: Dimension of the array that should be regularized 
- `sum_dims=[1, 2]`: A array containing the dimensions we want to sum over
- `weights=[1, 1]`: A array containing weights to weight the contribution of 
    different dimensions
- `step=1`: A integer indicating the step width for the array indexing
- `mode="forward"`: Either `"central"` or `"forward"` accounting for different
    modes of the spatial gradient. Default is "central".

# Examples
To create a regularizer for a 3D dataset where the third dimension
has a different sampling (factor 4 larger) than the first two dimensions.
For the spatial gradient `"forward"` is used.

```julia-repl
julia> GR(sum_dims=[1, 2, 3], weights=[1, 1, 0.25], mode="forward")
```
"""
function GR(; num_dim=5, sum_dims=[1, 2], weights=[1, 1], step=1,
              mode="central", ϵ=1e-14)
    if mode == "central"
        GRf = @eval arr -> ($(generate_GR(num_dim, sum_dims, weights,
                                        step, (-1) * step, ϵ)...))
    elseif mode == "forward"
        GRf = @eval arr -> ($(generate_GR(num_dim, sum_dims, weights,
                                        step, 0, ϵ)...))
    end
    

    return rec -> GRf(rec) / 4 / length(rec)
end


"""
    generate_TV(num_dim, sum_dims_arr, weights, ind1, ind2, ϵ)

Generate the Tullio statement for computing the Good's roughness.
`num_dim` is the dimension of the array. 
`sum_dims_arr` is a array
indicating over which dimensions we must sum over.
`weights` is a array of a weight for the different dimension.
`ind1` and `ind2` are the offsets for the difference.
`ϵ` is a numerical constant to prevent division by zero. 
    this is important for the gradient 
"""
function generate_TV(num_dim, sum_dims_arr, weights, ind1, ind2, ϵ=1e-8)
    out, add = [], []
    for (d, w) in zip(sum_dims_arr, weights)
        inds1, inds2 = generate_indices(num_dim, d, ind1, ind2) 
        push!(add, :(abs2($w * arr[$(inds1...)] - $w * arr[$(inds2...)])))
    end
    push!(add, ϵ)
    push!(out, :(@tullio res = sqrt(+($(add...)))))
    return out
end


"""
    TV(; <keyword arguments>)

This function returns a function to calculate the Total Variation regularizer
of a n-dimensional array. 

# Arguments
- `num_dim=5`: 
- `sum_dims=[1, 2, 3]`: A array containing the dimensions we want to sum over
- `weights=[1, 1, 0.25]`: A array containing weights to weight the contribution of 
    different dimensions
- `step=1`: A integer indicating the step width for the array indexing
- `mode="central"`: Either `"central"` or `"forward"` accounting for different
    modes of the spatial gradient. Default is "central".

# Examples
To create a regularizer for a 4D dataset where the third dimension
has a different sampling (factor 4 larger) than the first two dimensions.
For the spatial gradient `"forward"` is used. The fourth dimensions is not
considered in the regularizing process itself 
but just acts as a summation of all 3D regularizers.

```julia-repl
julia> GR(num_dim=3, sum_dims=[1, 2, 3], weights=[1, 1, 0.25], mode="forward")
```


# Bugs
Result is sometimes NaN -> current issue

"""
function TV(; num_dim=5, sum_dims=[1, 2], weights=[1, 1], step=1, mode="central")

    if mode == "central"
        total_var = @eval arr -> ($(generate_TV(num_dim, sum_dims, weights,
                                        step, (-1) * step)...))
    elseif mode == "forward"
        total_var = @eval arr -> ($(generate_TV(num_dim, sum_dims, weights,
                                        step, 0)...))
    end
    return rec -> total_var(rec) / length(rec)
end
