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
    HS(; p=1)

Hessian Schatten norm. `p` determines which Schatten norm is used.

This regularizer only works with 2D arrays at the moment.
"""
function HS(;p=1)
    if isone(p)
        return HS1
    end
    
    f(x) = HSp(x, p=p)
    return f
end

"""
Hessian schatten norm for p=1 efficiently with Tullio.
"""
function HS1(arr)
    H11 = Δr1r1(arr)
    H22 = Δr2r2(arr)
    return schatten_norm_1(H11, H22)
end

function schatten_norm_1(a, d)
    @tullio A[i, j] := a[i, j] + d[i, j]
    @tullio res = abs(1f-8 + A[i, j])
end

"""
Hessian schatten norm for p.
But not as fast as p=1
"""
function HSp(arr; p=1)
    H11 = Δr1r1(arr)
    H22 = Δr2r2(arr)
    H12 = Δr1r2(arr)
    
  
    res = schatten_norm_tullio(H11, H12, H22, p)
    return sum(res) 
end


function schatten_norm(H11, H12, H22, p)
    λ₁, λ₂ = eigvals_symmetric(H11, H12, H22) 
    return (λ₁^p + λ₂^p )^(1/p)
end

function schatten_norm_tullio(H11, H12, H22, p)
    λ₁, λ₂ = eigvals_symmetric_tullio(H11, H12, H22) 
    return @tullio res = (1f-8 + λ₁[i, j]^p + λ₂[i, j]^p)^(1/p)
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
