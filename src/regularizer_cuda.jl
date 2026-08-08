export TV_cuda, GR_cuda, TH_cuda, Tikhonov_cuda

f_inds(rs, b) = ntuple(i -> i == b ? rs[i] .+ 1 : rs[i], length(rs))

# offset a (possible subset of) dimensions by an arbitrary amount
shift_inds(rs, d, off) = ntuple(i -> i == d ? rs[i] .+ off : rs[i], length(rs))


"""
    TV_cuda(; num_dims=nothing, sum_dims=nothing, weights=nothing, step=1, mode="forward", ϵ=1f-8)
This function returns a function to calculate the Total Variation regularizer
of a n-dimensional array.

# Arguments
- `num_dims=nothing`: Dimension of the array that should be regularized. When `nothing`, it is inferred from the array upon use.
- `sum_dims=nothing`: A array containing the dimensions we want consider in the TV calculation. Defaults to all dimensions (`1:N`).
- `weights=nothing`: A array containing weights to weight the contribution of 
    different dimensions. If `weights=nothing` all dimensions are weighted equally.
- `step=1`: A integer indicating the step width for the array indexing
- `mode="forward"`: Either `"central"` or `"forward"` accounting for different
    modes of the spatial gradient. Default is "forward".
- `ϵ=1f-8`: A constant which allows to smoothly vary between TV and grad^2 regularization: L = sqrt.(grad^2+ϵ).

```julia-repl
julia> using CUDA

julia> reg = TV_cuda(num_dims=2);

julia> reg(CuArray([1 2 3; 4 5 6; 7 8 9]))
12.649111f0
```
"""
function TV_cuda(; num_dims=nothing, sum_dims=nothing, weights=nothing, step=1,
                 mode="forward", ϵ=1f-8)
    if !(mode in ("forward", "central"))
        throw(ArgumentError("The provided mode is not valid."))
    end
    if isnothing(num_dims)
        return arr -> TV_view(arr, sum_dims, weights, step, mode, ϵ)
    else
        s_dims = isnothing(sum_dims) ? collect(1:num_dims) : collect(sum_dims)
        ws = isnothing(weights) ? ones(Int, num_dims) : weights
        return arr -> TV_view(arr, s_dims, ws, step, mode, ϵ)
    end
    
    # if isnothing(num_dims) 
    #     return arr -> TV_view(arr, sum_dims, weights, step, mode, ϵ)
    #     # return arr -> TV_view(arr, weights, ϵ)
    # elseif num_dims == 3
    #     return arr -> TV_3D_view(arr, weights, ϵ)
    # elseif num_dims == 2
    #     return arr -> TV_2D_view(arr, weights, ϵ)
    # elseif num_dims == 1
    #     return arr -> TV_1D_view(arr, weights, ϵ)
    # else
    #     throw(ArgumentError("num_dims must be nothing or 2 or 3 "))
    # end
    # return reg_TV
end

function TV_view(arr::AbstractArray{T, N}, sum_dims=nothing, weights=nothing,
                 step=1, mode="forward", ϵ=1f-8) where {T, N}
    if isnothing(sum_dims)
        sum_dims = collect(1:N)
    end
    if isnothing(weights)
        weights = ones(Float32, N)
    end
    rs = ntuple(N) do d
        if d in sum_dims
            if mode == "forward"
                (first(axes(arr, d))):(last(axes(arr, d)) .- step)
            else
                (first(axes(arr, d)) .+ step):(last(axes(arr, d)) .- step)
            end
        else
            axes(arr, d)
        end
    end
    arr0 = view(arr, rs...)
    term = zero(arr0)
    if mode == "forward"
        for (d, w) in zip(sum_dims, weights)
            term = term .+ w .* (view(arr, shift_inds(rs, d, step)...) .- arr0).^2
        end
    else
        for (d, w) in zip(sum_dims, weights)
            term = term .+ w .* (view(arr, shift_inds(rs, d, step)...) .-
                                 view(arr, shift_inds(rs, d, -step)...)).^2
        end
    end
    return @fastmath sum(sqrt.(ϵ .+ term))
end

function TV_1D_view(arr::AbstractArray{T, N}, weights=nothing, ϵ=1f-8) where {T, N}
    if isnothing(weights)
        weights = ones(Float32, ndims(arr))
    end
    as = ntuple(i -> axes(arr, i), Val(N))
    rs = map(x -> first(x):last(x)-1, as)
    arr0 = view(arr, f_inds(rs, 0)...)
    arr1 = view(arr, f_inds(rs, 1)...)
    return @fastmath sum(sqrt.(ϵ .+ weights[1] .* (arr1 .- arr0).^2))
end

function TV_2D_view(arr::AbstractArray{T, N}, weights=nothing, ϵ=1f-8) where {T, N}
    if isnothing(weights)
        weights = ones(Float32, ndims(arr))
    end
    as = ntuple(i -> axes(arr, i), Val(N))
    rs = map(x -> first(x):last(x)-1, as)
    arr0 = view(arr, f_inds(rs, 0)...)
    arr1 = view(arr, f_inds(rs, 1)...)
    arr2 = view(arr, f_inds(rs, 2)...)
    return @fastmath sum(sqrt.(ϵ .+ weights[1] .* (arr1 .- arr0).^2 .+ weights[2] .* (arr0 .- arr2).^2))
end

function TV_3D_view(arr::AbstractArray{T, N}, weights=nothing, ϵ=1f-8) where {T, N}
    if isnothing(weights)
        weights = ones(Float32, ndims(arr))
    end
    as = ntuple(i -> axes(arr, i), Val(N))
    rs = map(x -> first(x):last(x)-1, as)
    arr0 = view(arr, f_inds(rs, 0)...)
    arr1 = view(arr, f_inds(rs, 1)...)
    arr2 = view(arr, f_inds(rs, 2)...)
    arr3 = view(arr, f_inds(rs, 3)...)
    return @fastmath sum(sqrt.(ϵ .+ weights[1] .* (arr1 .- arr0).^2 .+ 
                               weights[2] .* (arr2 .- arr0).^2 .+  weights[3] .* (arr3 .- arr0).^2 ))
end


"""
    GR_cuda(; num_dims=nothing, sum_dims=nothing, weights=nothing, step=1, mode="forward", ϵ=1f-8)

This function returns a function to calculate the Good's roughness regularizer
of a n-dimensional array on CUDA (or CPU) arrays.

Differentiable with `Zygote` on `CuArray`s because it avoids `Tullio` and only
uses `view`/broadcast/`sum` operations. The math is identical to
[`GR`](@ref) (see [`GR`](@ref) for a description of the arguments).

If `num_dims` is `nothing`, the number of dimensions is inferred from the
array upon use. The default weights then assume `1` for each of the `num_dims`
dimensions.
"""
function GR_cuda(; num_dims=nothing, sum_dims=nothing, weights=nothing, step=1,
                 mode="forward", ϵ=1f-8)
    if !(mode in ("forward", "central"))
        throw(ArgumentError("The provided mode is not valid."))
    end
    if isnothing(num_dims)
        if isnothing(sum_dims) && isnothing(weights)
            # cache the default per-rank `s_dims`/`weights` so that the common
            # auto path does not allocate on every call
            s_dims_all = ntuple(N -> collect(1:N), NMAX)
            ws_all = ntuple(N -> ones(Int, N), NMAX)
            return arr -> begin
                N = ndims(arr)
                if N <= NMAX
                    return GR_cuda_apply(arr, s_dims_all[N], ws_all[N], step, mode, ϵ)
                end
                return GR_cuda_apply(arr, collect(1:N), ones(Int, N), step, mode, ϵ)
            end
        end
        return arr -> begin
            N = ndims(arr)
            s_dims = isnothing(sum_dims) ? collect(1:N) : collect(sum_dims)
            ws = isnothing(weights) ? ones(Int, N) : weights
            GR_cuda_apply(arr, s_dims, ws, step, mode, ϵ)
        end
    else
        s_dims = isnothing(sum_dims) ? collect(1:num_dims) : collect(sum_dims)
        ws = isnothing(weights) ? ones(Int, num_dims) : weights
        return arr -> GR_cuda_apply(arr, s_dims, ws, step, mode, ϵ)
    end
end

function GR_cuda_apply(arr::AbstractArray{T, N}, s_dims, ws, step, mode, ϵ) where {T, N}
    a = sqrt.(arr .+ ϵ)
    rs = ntuple(N) do d
        if d in s_dims
            if mode == "forward"
                (first(axes(a, d))):(last(axes(a, d)) .- step)
            else
                (first(axes(a, d)) .+ step):(last(axes(a, d)) .- step)
            end
        else
            axes(a, d)
        end
    end
    a0 = view(a, rs...)
    term = zero(a0)
    if mode == "forward"
        for (d, w) in zip(s_dims, ws)
            term = term .+ w .* (view(a, shift_inds(rs, d, step)...) .+ a0)
        end
    else
        for (d, w) in zip(s_dims, ws)
            term = term .+ w .* (view(a, shift_inds(rs, d, step)...) .+
                                 view(a, shift_inds(rs, d, -step)...))
        end
    end
    prefactor = mode == "forward" ? -4 / step : -2 / step
    return prefactor * sum(a0 .* (term .- 2 * sum(ws) * a0))
end


"""
    TH_cuda(; num_dims=nothing, weights=nothing, ϵ=1f-8)

This function returns a function to calculate the Total Hessian norm
of a n-dimensional array on CUDA (or CPU) arrays.

Differentiable with `Zygote` on `CuArray`s. The math is identical to
[`TH`](@ref) (see [`TH`](@ref) for the arguments).
"""
function TH_cuda(; num_dims=nothing, weights=nothing, ϵ=1f-8)
    if isnothing(num_dims)
        return arr -> TH_view(arr, weights, ϵ)
    elseif num_dims == 1
        return arr -> TH_1D_view(arr, weights, ϵ)
    elseif num_dims == 2
        return arr -> TH_2D_view(arr, weights, ϵ)
    elseif num_dims == 3
        return arr -> TH_3D_view(arr, weights, ϵ)
    else
        throw(ArgumentError("num_dims must be 1, 2 or 3"))
    end
end

function TH_view(arr::AbstractArray{T, 1}, weights=nothing, ϵ=1f-8) where {T}
    return TH_1D_view(arr, weights, ϵ)
end

function TH_view(arr::AbstractArray{T, 2}, weights=nothing, ϵ=1f-8) where {T}
    return TH_2D_view(arr, weights, ϵ)
end

function TH_view(arr::AbstractArray{T, 3}, weights=nothing, ϵ=1f-8) where {T}
    return TH_3D_view(arr, weights, ϵ)
end

function TH_view(arr::AbstractArray{T, N}, weights=nothing, ϵ=1f-8) where {T, N}
    throw(ArgumentError("TH only supports 1, 2 or 3 dimensions, got an array of $N dimensions."))
end

function TH_1D_view(arr::AbstractArray{T, N}, weights=nothing, ϵ=1f-8) where {T, N}
    if isnothing(weights)
        weights = ones(Float32, 1)
    end
    rs = (first(axes(arr, 1)) .+ 1):(last(axes(arr, 1)) .- 1)
    a0 = view(arr, rs)
    am = view(arr, rs .- 1)
    ap = view(arr, rs .+ 1)
    return sum(sqrt.(ϵ .+ weights[1]^2 .* abs2.(ap .+ am .- 2 .* a0)))
end

function TH_2D_view(arr::AbstractArray{T, N}, weights=nothing, ϵ=1f-8) where {T, N}
    if isnothing(weights)
        weights = ones(Float32, 2)
    end
    rs = ntuple(i -> (first(axes(arr, i)) .+ 1):(last(axes(arr, i)) .- 1), 2)
    a00 = view(arr, rs[1], rs[2])
    a10 = view(arr, rs[1] .+ 1, rs[2])
    am0 = view(arr, rs[1] .- 1, rs[2])
    a01 = view(arr, rs[1], rs[2] .+ 1)
    a0m = view(arr, rs[1], rs[2] .- 1)
    a11 = view(arr, rs[1] .+ 1, rs[2] .+ 1)
    term = weights[1]^2 * abs2.(a10 .+ am0 .- 2 .* a00) .+
           weights[2]^2 * abs2.(a01 .+ a0m .- 2 .* a00) .+
           2 * weights[1] * weights[2] * abs2.(a11 .- a10 .- a01 .+ a00)
    return sum(sqrt.(ϵ .+ term))
end

function TH_3D_view(arr::AbstractArray{T, N}, weights=nothing, ϵ=1f-8) where {T, N}
    if isnothing(weights)
        weights = ones(Float32, 3)
    end
    rs = ntuple(i -> (first(axes(arr, i)) .+ 1):(last(axes(arr, i)) .- 1), 3)
    a000 = view(arr, rs[1], rs[2], rs[3])
    a100 = view(arr, rs[1] .+ 1, rs[2], rs[3])
    am00 = view(arr, rs[1] .- 1, rs[2], rs[3])
    a010 = view(arr, rs[1], rs[2] .+ 1, rs[3])
    a0m0 = view(arr, rs[1], rs[2] .- 1, rs[3])
    a001 = view(arr, rs[1], rs[2], rs[3] .+ 1)
    a00m = view(arr, rs[1], rs[2], rs[3] .- 1)
    a110 = view(arr, rs[1] .+ 1, rs[2] .+ 1, rs[3])
    a101 = view(arr, rs[1] .+ 1, rs[2], rs[3] .+ 1)
    a011 = view(arr, rs[1], rs[2] .+ 1, rs[3] .+ 1)
    term = weights[1]^2 * abs2.(a100 .+ am00 .- 2 .* a000) .+
           weights[2]^2 * abs2.(a010 .+ a0m0 .- 2 .* a000) .+
           weights[3]^2 * abs2.(a001 .+ a00m .- 2 .* a000) .+
           2 * weights[1] * weights[2] * abs2.(a110 .- a100 .- a010 .+ a000) .+
           2 * weights[1] * weights[3] * abs2.(a101 .- a100 .- a001 .+ a000) .+
           2 * weights[2] * weights[3] * abs2.(a011 .- a001 .- a010 .+ a000)
    return sum(sqrt.(ϵ .+ term))
end

function Tikhonov_cuda(; num_dims=nothing, sum_dims=nothing, weights=nothing,
                       step=1, mode="laplace")
    if !(mode in ("laplace", "spatial_grad_square", "identity"))
        throw(ArgumentError("The provided mode is not valid."))
    end
    if isnothing(num_dims)
        return arr -> Tikhonov_view(arr, sum_dims, weights, step, mode)
    else
        s_dims = isnothing(sum_dims) ? collect(1:num_dims) : collect(sum_dims)
        ws = isnothing(weights) ? ones(Int, num_dims) : weights
        return arr -> Tikhonov_view(arr, s_dims, ws, step, mode)
    end
end

function Tikhonov_view(arr::AbstractArray{T, N}, sum_dims=nothing, weights=nothing,
                       step=1, mode="laplace") where {T, N}
    if isnothing(sum_dims)
        sum_dims = collect(1:N)
    end
    if isnothing(weights)
        weights = ones(Float32, N)
    end
    if mode == "identity"
        return sum(abs2.(arr))
    end
    off = mode == "spatial_grad_square" ? step : 1
    rs = ntuple(N) do d
        if d in sum_dims
            (first(axes(arr, d)) .+ off):(last(axes(arr, d)) .- off)
        else
            axes(arr, d)
        end
    end
    a0 = view(arr, rs...)
    if mode == "laplace"
        term = -2 .* sum(weights) .* a0
        for (d, w) in zip(sum_dims, weights)
            term = term .+ w .* view(arr, shift_inds(rs, d, 1)...)
            term = term .+ w .* view(arr, shift_inds(rs, d, -1)...)
        end
        return sum(abs2.(term))
    elseif mode == "spatial_grad_square"
        term = zero(a0)
        for (d, w) in zip(sum_dims, weights)
            term = term .+ w .* abs2.(view(arr, shift_inds(rs, d, step)...) .-
                                       view(arr, shift_inds(rs, d, (-1) * step)...))
        end
        return sum(term)
    else
        throw(ArgumentError("The provided mode is not valid."))
    end
end
