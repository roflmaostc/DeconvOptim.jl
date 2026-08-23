# KernelAbstractions-based regularizers.
#
# Each regularizer is implemented as a single GPU/CPU kernel that computes the
# regularizer value AND its gradient in one launch, without AD.  The generic
# scaffolding (block tree reduction, padded launch, work-buffer handling,
# callable functor + ChainRules rrule) lives here so that further regularizers
# (GRK, TikhonovK, THK, ...) only need to define
#
#   * a mode struct <: KARegularizerMode   (holds weights/step/ϵ/...)
#   * `pixel_work!(A, dA, m, p)`            (per-pixel value + gradient writes)
#   * a `XK()` constructor returning `KARegFun(recipe, nothing, nothing)`
#
# Currently implemented: TVK() (Total Variation), with modes
#   * "forward"     : TV = Σ sqrt(Σ_d w_d (a[p+e_d·step]-a[p])² + ϵ)   (smoothed isotropic)
#   * "central"     : same but with a[p+e_d·step]-a[p-e_d·step]
#   * "anisotropic" : TV = Σ_d w_d Σ |a[p+e_d·step]-a[p]|             (per-direction L1)
# which mirror the options of the Tullio-based TV() in regularizer.jl.

import KernelAbstractions

# export TVK, ka_regularizer_and_grad!

using KernelAbstractions: @kernel, @index, @groupsize, @uniform, @localmem,
                          @synchronize, get_backend

abstract type KARegularizerMode end

# ---------------------------------------------------------------------------
# TV modes
# ---------------------------------------------------------------------------

struct KATVForward{N, T, M} <: KARegularizerMode
    weights::NTuple{N, T}
    sumdims::NTuple{M, Int}
    step::Int
    ϵ::T
end

struct KATVCentral{N, T, M} <: KARegularizerMode
    weights::NTuple{N, T}
    sumdims::NTuple{M, Int}
    step::Int
    ϵ::T
end

struct KATVAniso{N, T, M} <: KARegularizerMode
    weights::NTuple{N, T}
    sumdims::NTuple{M, Int}
    step::Int
    ϵ::T
end

# ---------------------------------------------------------------------------
# per-pixel helpers
# ---------------------------------------------------------------------------

# pixel coordinate shifted by `off` along dimension `d`
@inline function shift(A, p, d, off)
    return CartesianIndex(ntuple(k -> k == d ? p[k] + off : p[k], ndims(A)))
end

# forward difference along all summed dimensions at q (q must be in-domain)
@inline function forward_diffs(A, m, q)
    ntuple(length(m.sumdims)) do i
        d = m.sumdims[i]
        A[shift(A, q, d, m.step)] - A[q]
    end
end

# central difference along all summed dimensions at q (q must be in-domain)
@inline function central_diffs(A, m, q)
    ntuple(length(m.sumdims)) do i
        d = m.sumdims[i]
        A[shift(A, q, d, m.step)] - A[shift(A, q, d, -m.step)]
    end
end

# is q inside the loss domain (all summed dims have a valid stencil)?
@inline function indomain(m::Union{KATVForward, KATVAniso}, sz, q)
    for d in m.sumdims
        (q[d] < 1 || q[d] + m.step > sz[d]) && return false
    end
    return true
end

@inline function indomain(m::KATVCentral, sz, q)
    for d in m.sumdims
        (q[d] < 1 + m.step || q[d] + m.step > sz[d]) && return false
    end
    return true
end

# smoothed isotropic loss and its derivative wrt a[q] (forward differences):
#   l = sqrt(Σ w g² + ϵ),  ∂l/∂a[q] = -Σ w g / l
@inline function loss_and_selfgrad(m::KATVForward, gs)
    s2 = zero(typeof(m.ϵ))
    s1 = zero(typeof(m.ϵ))
    @inbounds for i in eachindex(gs)
        w = m.weights[m.sumdims[i]]
        gi = gs[i]
        s2 += w * gi * gi
        s1 += w * gi
    end
    r = sqrt(s2 + m.ϵ)
    return r, -s1 / r
end

# central difference loss only; the central stencil a[q+e_d]-a[q-e_d] does not
# depend on a[q], so ∂l/∂a[q] = 0 (no self gradient term)
@inline function loss_only(m::KATVCentral, gs)
    s2 = zero(typeof(m.ϵ))
    @inbounds for i in eachindex(gs)
        w = m.weights[m.sumdims[i]]
        gi = gs[i]
        s2 += w * gi * gi
    end
    return sqrt(s2 + m.ϵ)
end

# ---------------------------------------------------------------------------
# per-pixel value + gradient
# ---------------------------------------------------------------------------

# Each work item owns one pixel p: it contributes its own loss term (value)
# and accumulates into dA[p] the gradient from its own loss and from the
# losses of every stencil-neighbour pixel q whose loss depends on a[p]
# (gather, no atomics needed).
@inline function pixel_work!(A, dA, m::KATVForward, p)
    T = typeof(m.ϵ)
    sz = size(A)
    s = zero(T)
    dgrad = zero(T)
    if indomain(m, sz, p)
        gs = forward_diffs(A, m, p)
        l, dl = loss_and_selfgrad(m, gs)
        s += l
        dgrad += dl
    end
    @inbounds for i in eachindex(m.sumdims)
        d = m.sumdims[i]
        q = shift(A, p, d, -m.step)
        if indomain(m, sz, q)
            gs = forward_diffs(A, m, q)
            l, _ = loss_and_selfgrad(m, gs)
            dgrad += m.weights[d] * gs[i] / l
        end
    end
    @inbounds dA[p] = dgrad
    return s
end

@inline function pixel_work!(A, dA, m::KATVCentral, p)
    T = typeof(m.ϵ)
    sz = size(A)
    s = zero(T)
    dgrad = zero(T)
    if indomain(m, sz, p)
        gs = central_diffs(A, m, p)
        s += loss_only(m, gs)
    end
    @inbounds for i in eachindex(m.sumdims)
        d = m.sumdims[i]
        for off in (-m.step, m.step)
            q = shift(A, p, d, off)
            if indomain(m, sz, q)
                gs = central_diffs(A, m, q)
                l = loss_only(m, gs)
                # q = p - step e_d : p = q + step e_d  →  +w g_d / l
                # q = p + step e_d : p = q - step e_d  →  -w g_d / l
                dgrad += off < 0 ? m.weights[d] * gs[i] / l : -m.weights[d] * gs[i] / l
            end
        end
    end
    @inbounds dA[p] = dgrad
    return s
end

@inline function pixel_work!(A, dA, m::KATVAniso, p)
    T = typeof(m.ϵ)
    sz = size(A)
    s = zero(T)
    dgrad = zero(T)
    @inbounds for i in eachindex(m.sumdims)
        d = m.sumdims[i]
        if p[d] + m.step <= sz[d]
            gd = A[shift(A, p, d, m.step)] - A[p]
            w = m.weights[d]
            s += w * abs(gd)
            dgrad -= w * sign(gd)
        end
    end
    @inbounds for i in eachindex(m.sumdims)
        d = m.sumdims[i]
        q = shift(A, p, d, -m.step)
        if q[d] >= 1 && q[d] + m.step <= sz[d]
            gd = A[shift(A, q, d, m.step)] - A[q]
            dgrad += m.weights[d] * sign(gd)
        end
    end
    @inbounds dA[p] = dgrad
    return s
end

# ---------------------------------------------------------------------------
# generic reduction scaffold
# ---------------------------------------------------------------------------

const KA_WG = 256

@kernel unsafe_indices = true function ka_regularizer_kernel!(A::AbstractArray{T},
                                                              dA, partials, m) where {T}
    N = @uniform prod(@groupsize())
    lg = @uniform trailing_zeros(N)
    idx = @index(Global, Linear)
    li = @index(Local, Linear)
    gi = @index(Group, Linear)
    total = length(A)

    s = zero(typeof(m.ϵ))
    if idx <= total
        p = CartesianIndices(size(A))[idx]
        s = pixel_work!(A, dA, m, p)
    end

    sm = @localmem typeof(m.ϵ) (N,)
    sm[li] = idx <= total ? s : zero(typeof(m.ϵ))
    @synchronize()
    for level in 1:lg
        stride = N >> level
        if li <= stride
            sm[li] += sm[li + stride]
        end
        @synchronize()
    end
    if li == 1
        partials[gi] = sm[1]
    end
end

"""
    ka_regularizer_and_grad!(dA, A, mode; backend=nothing, partials=nothing) -> value

Compute the regularizer `value` for `mode <: KARegularizerMode` and its gradient
into `dA`, in a single kernel launch on the backend of `A` (or `backend`).
`partials` is an optional reusable work buffer of length `cld(length(A), 256)`.
"""
function ka_regularizer_and_grad!(dA, A, m::KARegularizerMode; backend=nothing, partials=nothing)
    backend === nothing && (backend = get_backend(A))
    Tc = typeof(m.ϵ)
    total = length(A)
    nblocks = cld(total, KA_WG)
    if partials === nothing || length(partials) < nblocks
        partials = similar(A, Tc, nblocks)
    end
    kern = ka_regularizer_kernel!(backend, KA_WG)
    kern(A, dA, partials, m; ndrange=(nblocks * KA_WG,), workgroupsize=KA_WG)
    KernelAbstractions.synchronize(backend)
    return sum(partials)
end

# ---------------------------------------------------------------------------
# callable functor + Zygote/ChainRules integration
# ---------------------------------------------------------------------------

mutable struct KARegFun{R}
    recipe::R
    dA::Any
    partials::Any
end

function _ensure_buffers!(f::KARegFun, A, m)
    Tc = typeof(m.ϵ)
    if f.dA === nothing || size(f.dA) != size(A) ||
       eltype(f.dA) != eltype(A) || get_backend(f.dA) != get_backend(A)
        f.dA = similar(A)
    end
    nblocks = cld(length(A), KA_WG)
    if f.partials === nothing || length(f.partials) < nblocks ||
       eltype(f.partials) != Tc || get_backend(f.partials) != get_backend(A)
        f.partials = similar(A, Tc, nblocks)
    end
    return nothing
end

(f::KARegFun)(A::AbstractArray) = begin
    m = f.recipe(A)
    return _tvk_value_grad!(f, A, m)
end

# value + gradient in one call, via the single KA kernel on any backend
function _tvk_value_grad!(f::KARegFun, A, m)
    _ensure_buffers!(f, A, m)
    return ka_regularizer_and_grad!(f.dA, A, m; partials=f.partials)
end

function ChainRulesCore.rrule(f::KARegFun, A::AbstractArray)
    m = f.recipe(A)
    y = _tvk_value_grad!(f, A, m)
    function ka_regularizer_pullback(Δ)
        _tvk_value_grad!(f, A, m)
        return ChainRulesCore.NoTangent(), Δ .* f.dA
    end
    return y, ka_regularizer_pullback
end

# ---------------------------------------------------------------------------
# TVK
# ---------------------------------------------------------------------------

"""
    TVK(; num_dims=nothing, sum_dims=nothing, weights=nothing, step=1,
         mode="forward", ϵ=1f-8)

Like [`TV`](@ref), but implemented with a single KernelAbstractions kernel that
computes the value and its gradient at once (no Zygote/Tullio in the hot loop).
Runs on CPU and CUDA arrays.

`mode` is one of
  * `"forward"`     — smoothed isotropic TV with forward differences:
                      Σ sqrt(Σ_d w_d (a[p+e_d·step]-a[p])² + ϵ)
  * `"central"`     — same with central differences a[p+e_d·step]-a[p-e_d·step]
  * `"anisotropic"` — Σ_d w_d Σ |a[p+e_d·step]-a[p]| (per-direction L1)

Returns a callable object `f` with `f(a)` giving the scalar TV value.
Differentiable with Zygote (via a ChainRules rrule that reuses the same kernel
for the gradient).  For a GPU-resident solver, call
`ka_regularizer_and_grad!(dA, a, mode)` directly with preallocated buffers.
"""
function TVK(; num_dims=nothing, sum_dims=nothing, weights=nothing, step=1,
             mode="forward", ϵ=1f-8)
    mode in ("forward", "central", "anisotropic") ||
        throw(ArgumentError("mode must be \"forward\", \"central\" or " *
                            "\"anisotropic\", got $mode"))
    step >= 1 || throw(ArgumentError("step must be >= 1, got $step"))
    sd = isnothing(sum_dims) ? nothing : Tuple(Int.(collect(sum_dims)))
    if !isnothing(num_dims) && !isnothing(sd)
        all(d -> 1 <= d <= num_dims, sd) ||
            throw(ArgumentError("sum_dims entries out of range for num_dims=$num_dims"))
    end
    recipe = A -> _make_tvk_mode(A, num_dims, sd, weights, step, mode, ϵ)
    return KARegFun(recipe, nothing, nothing)
end

function _make_tvk_mode(A, num_dims, sd, weights, step, mode, ϵ)
    N = isnothing(num_dims) ? ndims(A) : num_dims
    sdT = isnothing(sd) ? ntuple(identity, N) : sd
    if !isnothing(weights) && length(weights) != length(sdT)
        throw(ArgumentError("length(weights) must equal the number of summed dimensions"))
    end
    Tc = promote_type(eltype(A), typeof(ϵ), isnothing(weights) ? Bool : eltype(weights))
    ws = ntuple(N) do d
        if d in sdT
            if isnothing(weights)
                one(Tc)
            else
                convert(Tc, weights[findfirst(==(d), sdT)])
            end
        else
            zero(Tc)
        end
    end
    if mode == "forward"
        return KATVForward(ws, sdT, Int(step), convert(Tc, ϵ))
    elseif mode == "central"
        return KATVCentral(ws, sdT, Int(step), convert(Tc, ϵ))
    else
        return KATVAniso(ws, sdT, Int(step), convert(Tc, ϵ))
    end
end
