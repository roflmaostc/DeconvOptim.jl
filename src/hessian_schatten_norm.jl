export HS

# References:
# * Lefkimmiatis, Stamatios, John Paul Ward, and Michael Unser. "Hessian Schatten-norm regularization for linear inverse problems." IEEE transactions on image processing 22.5 (2013): 1873-1888.
# * Lefkimmiatis, Stamatios, and Michael Unser. "Poisson image reconstruction with Hessian Schatten-norm regularization." IEEE transactions on image processing 22.11 (2013): 4314-4327.

function Δr1r1(x)
    return @tullio res[i, j] := x[i+2, j+0] - 2 * x[i+1, j] + x[i, j] (i in 1:size(x)[1]-2, j in 1:size(x)[2]-2)
end

function Δr2r2(x)
    return @tullio res[i, j] := x[i+0, j+2] - 2 * x[i, j+1] + x[i, j] (i in 1:size(x)[1]-2, j in 1:size(x)[2]-2)
end

function Δr1r2(x)
    return @tullio res[i, j] := x[i+1, j+1] - x[i+1, j] - x[i, j+1]  + x[i, j] (i in 1:size(x)[1]-2, j in 1:size(x)[2]-2)
end


"""
    HS(; p=1, sum_dims=nothing, weights=nothing)

Hessian Schatten norm. `p` determines which Schatten norm is used.

The Hessian is evaluated over exactly two array dimensions, which are given
by `sum_dims` (defaulting to `(1, 2)`). The selected dimensions may be a
subset of the array dimensions, so e.g. `sum_dims=(1, 2)` works for both 2D
and 3D (or higher dimensional) arrays -- the remaining dimensions only take
part in the final summation. Because the two rows and columns of the 2×2
pixel-wise Hessian are inverted in closed form for speed, `sum_dims` is
limited to exactly two dimensions; passing more throws an `ArgumentError`.

`weights[k]` pair positionally with `sum_dims[k]`: the diagonal Hessian entry
of dimension `d` enters with `weights[k]^2` and the cross entry between the
two dimensions enters with `2 * weights[k] * weights[l]`.

The computation is regularized by an internal smoothing constant `1f-8` (see
`ϵ` in `TV`/`TH`), so that the norm is differentiable.

# Arguments
- `p=1`: The order `p` of the Schatten norm. `p=1` uses a cheap fast path.
- `sum_dims=nothing`: The two array dimensions over which the Hessian is
    evaluated. Defaults to `(1, 2)`. Exactly two dimensions are supported.
- `weights=nothing`: Weights for the different dimensions, matched
    positionally to `sum_dims`. If `nothing` all dimensions are weighted
    equally.
"""
function HS(; p=1, sum_dims=nothing, weights=nothing)
    if isone(p)
        return arr -> HS1(arr, sum_dims=sum_dims, weights=weights)
    end
    return arr -> HSp(arr, p=p, sum_dims=sum_dims, weights=weights)
end

"""
Hessian schatten norm for p=1 efficiently with Tullio.
"""
function HS1(arr; sum_dims=nothing, weights=nothing)
    if isnothing(sum_dims) && isnothing(weights) && ndims(arr) == 2
        H11 = Δr1r1(arr)
        H22 = Δr2r2(arr)
        return schatten_norm_1(H11, H22)
    end
    d1, d2 = hs_validate_sum_dims(arr, sum_dims)
    w = hs_weights(weights)
    H11 = hs_diag(arr, d1, (d1, d2))
    H22 = hs_diag(arr, d2, (d1, d2))
    w1sq = w[1] * w[1]
    w2sq = w[2] * w[2]
    trace = @~ w1sq .* H11 .+ w2sq .* H22
    return @fastmath sum(abs.(1f-8 .+ trace))
end

function schatten_norm_1(a, d)
    @tullio A[i, j] := a[i, j] + d[i, j]
    @tullio res = abs(1f-8 + A[i, j])
end

"""
Hessian schatten norm for p.
But not as fast as p=1
"""
function HSp(arr; p=1, sum_dims=nothing, weights=nothing)
    if isnothing(sum_dims) && isnothing(weights) && ndims(arr) == 2
        H11 = Δr1r1(arr)
        H22 = Δr2r2(arr)
        H12 = Δr1r2(arr)
        return sum(schatten_norm_tullio(H11, H12, H22, p))
    end
    d1, d2 = hs_validate_sum_dims(arr, sum_dims)
    w = hs_weights(weights)
    H11 = hs_diag(arr, d1, (d1, d2))
    H22 = hs_diag(arr, d2, (d1, d2))
    H12 = hs_cross(arr, d1, d2)
    w1sq = w[1] * w[1]
    w2sq = w[2] * w[2]
    a = @~ w1sq .* H11
    d = @~ w2sq .* H22
    b = @~ (w[1] * w[2]) .* H12
    λ1, λ2 = hs_eigvals(a, b, d)
    return @fastmath sum(abs.((1f-8 .+ λ1.^p .+ λ2.^p)).^(1 / p))
end


# ---------------------------------------------------------------------------
# generalized Hessian components for an arbitrary number of array dimensions,
# with the Hessian evaluated over exactly two dimensions `s_dims`.
# ---------------------------------------------------------------------------

# validate `sum_dims` against the array; exactly two distinct dims are allowed
function hs_validate_sum_dims(arr, sum_dims)
    N = ndims(arr)
    if isnothing(sum_dims)
        d = (1, 2)
    else
        d = Tuple(Int.(collect(sum_dims)))
    end
    if length(d) != 2
        throw(ArgumentError("HS only supports exactly two `sum_dims` (the 2x2 pixel-wise Hessian is inverted in closed form), got $(length(d)): $d. Use e.g. `sum_dims=(1, 2)`."))
    end
    if d[1] == d[2]
        throw(ArgumentError("HS `sum_dims` must be two distinct dimensions, got $d"))
    end
    if any(x -> x < 1 || x > N, d)
        throw(ArgumentError("HS `sum_dims` $d out of range for an array with $N dimensions"))
    end
    return d
end

# positional weights aligned to `sum_dims`
function hs_weights(weights)
    if isnothing(weights)
        return ones(Float32, 2)
    elseif length(weights) >= 2
        return collect(Float32, weights[1:2])
    else
        return vcat(Float32.(weights), ones(Float32, 2 - length(weights)))
    end
end

# base index ranges: dims in `s_dims` are cropped by 2 (the finite-difference
# stencils need two neighbours), all other dims keep their full range
function hs_ranges(arr, s_dims)
    return ntuple(i -> i in s_dims ? Base.OneTo(size(arr, i) - 2) : axes(arr, i), ndims(arr))
end

# apply integer offsets to a (subset of) dims of an index range tuple
function hs_offs(rs, offs::Dict{Int,Int})
    return ntuple(i -> haskey(offs, i) ? (rs[i] .+ offs[i]) : rs[i], length(rs))
end

# second derivative along dim `d`: x[i+2] - 2 x[i+1] + x[i]
function hs_diag(arr, d, s_dims)
    rs = hs_ranges(arr, s_dims)
    a0 = view(arr, rs...)
    a1 = view(arr, hs_offs(rs, Dict(d => 1))...)
    a2 = view(arr, hs_offs(rs, Dict(d => 2))...)
    return a2 .- 2 .* a1 .+ a0
end

# mixed derivative along dims `d` and `e`: x[i+1,j+1] - x[i+1,j] - x[i,j+1] + x[i,j]
function hs_cross(arr, d, e)
    rs = hs_ranges(arr, (d, e))
    a00 = view(arr, rs...)
    a10 = view(arr, hs_offs(rs, Dict(d => 1))...)
    a01 = view(arr, hs_offs(rs, Dict(e => 1))...)
    a11 = view(arr, hs_offs(rs, Dict(d => 1, e => 1))...)
    return a11 .- a10 .- a01 .+ a00
end

# eigenvalues of the weighted 2x2 pixel-wise Hessian [[a, b], [b, d]]
function hs_eigvals(a, b, d)
    A = a .+ d
    B = sqrt.(1f-8 .+ (a .- d).^2 .+ 4 .* b.^2)
    λ1 = @~ 0.5 .* (A .+ B)
    λ2 = @~ 0.5 .* (A .- B)
    return λ1, λ2
end


function schatten_norm(H11, H12, H22, p)
    λ₁, λ₂ = eigvals_symmetric(H11, H12, H22)
    return (λ₁^p + λ₂^p )^(1/p)
end

function schatten_norm_tullio(H11, H12, H22, p)
    λ₁, λ₂ = eigvals_symmetric_tullio(H11, H12, H22)
    return @tullio res = abs((1f-8 + λ₁[i, j]^p + λ₂[i, j]^p))^(1/p)
end

"""
    eigvals_symmetric(a,b,c)

Calculate the eigenvalues of the matrix
[a b; b d] analytically.
"""
function eigvals_symmetric(a, b, d)
    A = a+d
    B = sqrt((a-d)^2+4*b^2)
    λ₁ = 0.5 * (A + B)
    λ₂ = 0.5 * (A - B)
    return λ₁, λ₂
end

function eigvals_symmetric_tullio(a, b, d)
    @tullio A[i, j] := a[i, j] + d[i, j]
    @tullio B[i, j] := sqrt(1f-8 + (a[i, j]-d[i, j])^2+4*b[i, j]^2)
    @tullio λ₁[i, j] := 0.5 * (A[i, j] + B[i, j])
    @tullio λ₂[i, j] := 0.5 * (A[i, j] - B[i, j])
    return λ₁, λ₂
end