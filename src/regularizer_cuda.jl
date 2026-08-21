f_inds(rs, b) = ntuple(i -> i == b ? rs[i] .+ 1 : rs[i], length(rs))

# offset a (possible subset of) dimensions by an arbitrary amount
shift_inds(rs, d, off) = ntuple(i -> i == d ? rs[i] .+ off : rs[i], length(rs))

macro lazybc(ex)
    esc(_lazybc(ex))
end

function _lazybc(ex)
    ex isa Expr || return ex

    # f.(args...)
    if ex.head === :. && length(ex.args) == 2
        f, args = ex.args

        return Expr(
            :call,
            :(Base.Broadcast.broadcasted),
            _lazybc(f),
            (_lazybc(a) for a in args.args)...,
        )
    end

    # .+, .-, .*, ./, etc.
    if ex.head === :call
        f = ex.args[1]

        if f isa Symbol && startswith(String(f), ".")
            op = Symbol(String(f)[2:end])

            return Expr(
                :call,
                :(Base.Broadcast.broadcasted),
                op,
                (_lazybc(a) for a in ex.args[2:end])...,
            )
        end

        return Expr(:call, map(_lazybc, ex.args)...)
    end

    return Expr(ex.head, map(_lazybc, ex.args)...)
end

# Clean, one-line convenience function
# sumbc(bc::Base.Broadcast.Broadcasted) = sum(bc[I] for I in CartesianIndices(Base.Broadcast.instantiate(bc)))

# Or using mapreduce (which the compiler optimizes identically):
# sumbc(bc::Base.Broadcast.Broadcasted) = let bci = Base.Broadcast.instantiate(bc)
#     mapreduce(I -> bci[I], +, CartesianIndices(bci))
# end

function sumbc(bc::Base.Broadcast.Broadcasted)
    bci = Base.Broadcast.instantiate(bc)    
    return sum(bci)    
end

sumbc(a::AbstractArray) = sum(a)

# Custom rule for Zygote / ChainRules-compatible ADs
# function ChainRulesCore.rrule(::typeof(sumbc), bc::Base.Broadcast.Broadcasted)
#     y = sumbc(bc)
    
#     function sumbc_pullback(Δy)
#     @show "in pullback"
#         # The gradient of sum(f(x)) w.r.t f(x) is just 1.0 * Δy broadcasted across the structure.
#         # We materialise the broadcasted gradient expression lazy tree * Δy:
#         # Base.Broadcast.broadcasted(*, Δy, ...) 
#         # Or leverage Julia's native broadcast pullback mechanism:
        
#         # For simple broadcast trees, the gradient w.r.t the unmaterialized tree 
#         # propagates Δy back into the leaves using Base.Broadcast:
#         return (NoTangent(), Base.Broadcast.broadcasted(*, Δy, 1.0)) 
#     end
    
#     return y, sumbc_pullback
# end

"""
    HS_cuda(; p=1, sum_dims=nothing, weights=nothing)

This function returns a function to calculate the Hessian Schatten norm
of an n-dimensional array on CUDA (or CPU) arrays.

Differentiable with `Zygote` on `CuArray`s because it avoids `Tullio` and only
uses `view`/broadcast/`sum` operations. The math is identical to
[`HS`](@ref) (see [`HS`](@ref) for a description of the arguments).
"""
function HS_cuda(; p=1, sum_dims=nothing, weights=nothing)
    return arr -> HS_generic(arr, p, sum_dims, weights)
end


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
    # NOTE: use `abs2` (not `.^2`) here. `@lazybc` builds a deeply nested lazy
    # `Broadcasted` tree, and Zygote's CUDA backward pass emits NaN when
    # differentiating a `broadcasted(^, x, 2)` (an integer-exponent `.^2`) deep
    # inside such a tree. `abs2` has a well-defined GPU chain rule, so the
    # gradient stays finite. For real inputs `abs2(x) == x^2` (derivative `2x`),
    # so the math is identical.
    # NOTE: we accumulate `term` without a materialized `zero(arr0)` seed. The
    # first dimension's term is built directly and the rest are folded in, so
    # the whole regularizer stays a single lazy `Broadcasted` tree that `sumbc`
    # reduces in one fused pass (no full-size intermediate array is allocated).
    d0, w0 = first(zip(sum_dims, weights))
    if mode == "forward"
        term = @lazybc (w0 .* abs2.(view(arr, shift_inds(rs, d0, step)...) .- arr0))
        for (d, w) in zip(sum_dims[2:end], weights[2:end])
            term = @lazybc (term .+ w .* abs2.(view(arr, shift_inds(rs, d, step)...) .- arr0))
        end
    else
        term = @lazybc (w0 .* abs2.(view(arr, shift_inds(rs, d0, step)...) .-
                             view(arr, shift_inds(rs, d0, -step)...)))
        for (d, w) in zip(sum_dims[2:end], weights[2:end])
            term = @lazybc (term .+ w .* abs2.(view(arr, shift_inds(rs, d, step)...) .-
                                 view(arr, shift_inds(rs, d, -step)...)))
        end
    end
    expr = @lazybc sqrt.(ϵ .+ term)
    return @fastmath sumbc(expr)
    # br_exp = Broadcast.instantiate(Broadcast.broadcasted(absdif, (@view a[2:end,:]), (@view a[1:end-1,:])));
    # expr = @lazybc sqrt.(ϵ .+ term)
    # return @fastmath sumbc()
end

function TV_1D_view(arr::AbstractArray{T, N}, weights=nothing, ϵ=1f-8) where {T, N}
    if isnothing(weights)
        weights = ones(Float32, ndims(arr))
    end
    as = ntuple(i -> axes(arr, i), Val(N))
    rs = map(x -> first(x):last(x)-1, as)
    arr0 = view(arr, f_inds(rs, 0)...)
    arr1 = view(arr, f_inds(rs, 1)...)
    # `abs2` instead of `.^2`: see the comment in `TV_view` (Zygote CUDA
    # backward produces NaN for a lazy `.^2` inside a nested `Broadcasted`).
    expr = @lazybc sqrt.(ϵ .+ weights[1] .* abs2.(arr1 .- arr0))
    return @fastmath sumbc(expr)
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
    # `abs2` instead of `.^2`: see the comment in `TV_view` (Zygote CUDA
    # backward produces NaN for a lazy `.^2` inside a nested `Broadcasted`).
    expr = @lazybc sqrt.(ϵ .+ weights[1] .* abs2.(arr1 .- arr0) .+ weights[2] .* abs2.(arr0 .- arr2))
    return @fastmath sumbc()
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
    # `abs2` instead of `.^2`: see the comment in `TV_view` (Zygote CUDA
    # backward produces NaN for a lazy `.^2` inside a nested `Broadcasted`).
    expr = @lazybc sqrt.(ϵ .+ weights[1] .* abs2.(arr1 .- arr0) .+ 
                               weights[2] .* abs2.(arr2 .- arr0) .+  weights[3] .* abs2.(arr3 .- arr0) )
    return @fastmath sumbc(expr)
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
    rs = ntuple(N) do d
        if d in s_dims
            if mode == "forward"
                (first(axes(arr, d))):(last(axes(arr, d)) .- step)
            else
                (first(axes(arr, d)) .+ step):(last(axes(arr, d)) .- step)
            end
        else
            axes(arr, d)
        end
    end
    # Keep `sqrt.(arr .+ ϵ)` lazy (a `Broadcasted` over views) instead of
    # materialising it, and accumulate `term` without a `zero` seed, so the
    # whole regularizer is one fused `Broadcasted` tree that `sumbc` reduces
    # without allocating any full-size intermediate array.
    a0 = @lazybc sqrt.(view(arr, rs...) .+ ϵ)
    d0, w0 = first(zip(s_dims, ws))
    if mode == "forward"
        a0d = @lazybc sqrt.(view(arr, shift_inds(rs, d0, step)...) .+ ϵ)
        term = @lazybc (w0 .* (a0d .+ a0))
        for (d, w) in zip(s_dims[2:end], ws[2:end])
            ad = @lazybc sqrt.(view(arr, shift_inds(rs, d, step)...) .+ ϵ)
            term = @lazybc (term .+ w .* (ad .+ a0))
        end
    else
        ad = @lazybc sqrt.(view(arr, shift_inds(rs, d0, step)...) .+ ϵ)
        amd = @lazybc sqrt.(view(arr, shift_inds(rs, d0, -step)...) .+ ϵ)
        term = @lazybc (w0 .* (ad .+ amd))
        for (d, w) in zip(s_dims[2:end], ws[2:end])
            ad = @lazybc sqrt.(view(arr, shift_inds(rs, d, step)...) .+ ϵ)
            amd = @lazybc sqrt.(view(arr, shift_inds(rs, d, -step)...) .+ ϵ)
            term = @lazybc (term .+ w .* (ad .+ amd))
        end
    end
    prefactor = mode == "forward" ? -4 / step : -2 / step
    sw = sum(ws)
    expr = @lazybc (a0 .* (term .- ((2 * sw) .* a0)))
    return @fastmath prefactor * sumbc(expr) # fused the sum with the broadcast
end


"""
    TH_cuda(; num_dims=nothing, sum_dims=nothing, weights=nothing, ϵ=1f-8)

This function returns a function to calculate the Total Hessian norm
of a n-dimensional array on CUDA (or CPU) arrays.

Differentiable with `Zygote` on `CuArray`s. The math is identical to
[`TH`](@ref) (see [`TH`](@ref) for the arguments).

When `sum_dims` is given, the Hessian is only computed over the listed
dimensions (the remaining dimensions are summed over); the generic
`TH_view` fallback is used then.
"""
function TH_cuda(; num_dims=nothing, sum_dims=nothing, weights=nothing, ϵ=1f-8)
    if isnothing(sum_dims)
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
    else
        s_dims = collect(sum_dims)
        if isnothing(num_dims)
            return arr -> TH_view_generic(arr, s_dims, weights, ϵ)
        elseif num_dims <= 3
            if any(d -> d > num_dims, s_dims)
                throw(ArgumentError("sum_dims=$s_dims out of range for num_dims=$num_dims"))
            end
            return arr -> TH_view_generic(arr, s_dims, weights, ϵ)
        else
            throw(ArgumentError("num_dims must be 1, 2 or 3"))
        end
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

# generic TH over a subset of dimensions `sum_dims` (any rank, CPU + GPU).
# `weights[k]` pairs positionally with `sum_dims[k]`; dims not listed are only
# summed over. The stencil matches the `@tullio` CPU closure generated by
# `make_th_closure_generic` in `regularizer.jl`.
function TH_view_generic(arr::AbstractArray{T, N}, sum_dims, weights=nothing, ϵ=1f-8) where {T, N}
    s_dims = collect(sum_dims)
    if any(d -> d > N, s_dims)
        throw(ArgumentError("sum_dims=$(s_dims) out of range for an array with $N dimensions."))
    end
    if isnothing(weights)
        weights = ones(Float32, length(s_dims))
    else
        weights = collect(Float32, weights)
        if length(weights) < length(s_dims)
            weights = vcat(weights, ones(Float32, length(s_dims) - length(weights)))
        end
    end
    # only the dimensions in `s_dims` get cropped by 1 (the finite-difference
    # stencils take one neighbour on each side); all other dimensions keep
    # their full range and are only summed over (matching the `@tullio`
    # reduction in `make_th_closure_generic`).
    axes_all = ntuple(i -> axes(arr, i), N)
    rs = ntuple(N) do i
        if i in s_dims
            (first(axes_all[i]) .+ 1):(last(axes_all[i]) .- 1)
        else
            axes_all[i]
        end
    end

    term = @lazybc (0 .* view(arr, rs...))
    for (k, d) in enumerate(s_dims)
        pref = weights[k] * weights[k]
        ap = view(arr, shift_inds(rs, d, 1)...)
        am = view(arr, shift_inds(rs, d, -1)...)
        a0 = view(arr, rs...)
        term = @lazybc (term .+ pref .* abs2.(ap .+ am .- 2 .* a0))
    end
    for k in 1:length(s_dims), l in (k+1):length(s_dims)
        d, e = s_dims[k], s_dims[l]
        pref = 2 * weights[k] * weights[l]
        app = view(arr, shift_inds(f_inds(rs, e), d, 1)...)
        a0p = view(arr, f_inds(rs, e)...)
        ap0 = view(arr, f_inds(rs, d)...)
        a00 = view(arr, rs...)
        term = @lazybc (term .+ pref .* abs2.(app .- a0p .- ap0 .+ a00))
    end
    expr = @lazybc (sqrt.(ϵ .+ term))
    return @fastmath sumbc(expr)
end

function TH_1D_view(arr::AbstractArray{T, N}, weights=nothing, ϵ=1f-8) where {T, N}
    if isnothing(weights)
        weights = ones(Float32, 1)
    end
    rs = (first(axes(arr, 1)) .+ 1):(last(axes(arr, 1)) .- 1)
    a0 = view(arr, rs)
    am = view(arr, rs .- 1)
    ap = view(arr, rs .+ 1)
    expr = @lazybc (sqrt.(ϵ .+ (weights[1]*weights[1]) .* abs2.(ap .+ am .- 2 .* a0)))
    return @fastmath sumbc(expr) # fused the sum with the broadcast
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
    w11 = weights[1]*weights[1];
    w22 = weights[2]*weights[2];
    w12 = 2*weights[1]*weights[2];
    term = @lazybc (w11 .* abs2.(a10 .+ am0 .- 2 .* a00) .+ 
               w22 .* abs2.(a01 .+ a0m .- 2 .* a00)  .+
               w12 .* abs2.(a11 .- a10 .- a01 .+ a00));
    expr = @lazybc (sqrt.(ϵ .+ term))
    return @fastmath sumbc(expr) # fused the sum with the broadcast
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
    w11 = weights[1]*weights[1];
    w22 = weights[2]*weights[2];
    w33 = weights[3]*weights[3];
    w12 = 2*weights[1]*weights[2];
    w13 = 2*weights[1]*weights[3];
    w23 = 2*weights[2]*weights[3];
    term = @lazybc (w11 .* abs2.(a100 .+ am00 .- 2 .* a000) .+
           w22 .* abs2.(a010 .+ a0m0 .- 2 .* a000) .+
           w33 .* abs2.(a001 .+ a00m .- 2 .* a000) .+
           w12 .* abs2.(a110 .- a100 .- a010 .+ a000) .+
           w13 .* abs2.(a101 .- a100 .- a001 .+ a000) .+
           w23 .* abs2.(a011 .- a001 .- a010 .+ a000))
    expr = @lazybc (sqrt.(ϵ .+ term))
    return @fastmath sumbc(expr)  # fused the sum with the broadcast
end

"""
    Tikhonov_cuda(; num_dims=nothing, sum_dims=nothing, weights=nothing, step=1, mode="laplace")

This function returns a function to calculate the Tikhonov regularizer
of an n-dimensional array on CUDA (or CPU) arrays.

Differentiable with `Zygote` on `CuArray`s because it avoids `Tullio` and only
uses `view`/broadcast/`sum` operations. The math is identical to
[`Tikhonov`](@ref) (see [`Tikhonov`](@ref) for a description of the arguments).
"""
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
        expr = @lazybc abs2.(arr)
        return sumbc(expr)
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
        sw = -2 .* sum(weights)
        term = @lazybc sw .* a0
        for (d, w) in zip(sum_dims, weights)
            term = @lazybc (term .+ w .* view(arr, shift_inds(rs, d, 1)...))
            term = @lazybc (term .+ w .* view(arr, shift_inds(rs, d, -1)...))
        end
        expr = @lazybc abs2.(term)
        return @fastmath sumbc(expr)
    elseif mode == "spatial_grad_square"
        term = zero(a0)
        for (d, w) in zip(sum_dims, weights)
            term = @lazybc (term .+ w .* abs2.(view(arr, shift_inds(rs, d, step)...) .-
                                       view(arr, shift_inds(rs, d, (-1) * step)...)))
        end
        return @fastmath sumbc(term)
    else
        throw(ArgumentError("The provided mode is not valid."))
    end
end
