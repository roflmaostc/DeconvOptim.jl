# Regularizer

Regularizer are commonly used in inverse problems and especially in deconvolution to obtain solutions which are optimal with respect to some prior. 
We have included five common regularizers: Good's Roughness, Total Variation, Tikhonov, the Total Hessian norm and the Hessian Schatten norm.
The regularizer take the current reconstruction $S(r)$ as argument and return a scalar value. This value should be also minimized and is also added
to the loss function.
Each regularizer produces some characteristic image styles.

## Common arguments

All regularizers accept the following keyword arguments:

- `sum_dims`: An array (or tuple) containing the array dimensions over which the regularizer is computed. The remaining dimensions are only summed over. This allows e.g. `sum_dims=[1, 2]` on a 3D array to regularize only in the lateral directions while the third dimension just contributes to the sum. For [`HS()`](@ref) exactly two `sum_dims` must be given.
- `weights`: An array of weights matched positionally to `sum_dims`, i.e. `weights[k]` belongs to the dimension `sum_dims[k]`. A diagonal term of dimension `i` enters with `weights[i]^2` and a cross term between dimensions `i` and `j` enters with `2 * weights[i] * weights[j]`. If `weights=nothing` all dimensions are weighted equally.
- `num_dims`: The number of spatial dimensions of the array. When `nothing` (the default) it is inferred from the array upon use.

When applied to a `CuArray`, [`TV()`](@ref), [`GR()`](@ref), [`TH()`](@ref) and [`Tikhonov()`](@ref) automatically dispatch to GPU compliant view/broadcast based implementations.


# Good's Roughness (GR)
The Good's roughness definition was taken from [Good:71](@cite) and [Verveer:98](@cite).
For Good's roughness several identical expressions can be derived. We implemented the following one:

$\text{Reg}(S(r)) = \sum_r \sqrt{S(r)} (\Delta_N \sqrt{S})(r)$

where $N$ is the dimension of $S(r)$. $\sqrt S$ is applied elementwise.
$\Delta_n \sqrt{S(r)}$ is the n-dimensional discrete Laplace operator. As 2D example where $r = (x,y)$:

$(\Delta_n \sqrt{S})(r) = \frac{\sqrt{S(x + s_x, y)} + \sqrt{S(x - s_x, y)} + \sqrt{S(x, y+s_y)} + \sqrt{S(x, y-s_y)} - 4 \cdot \sqrt{S(x, y)}}{s_x \cdot s_y}$

where $s_x$ and $s_y$ are the stencil width in the respective dimension. The Laplace operator can be straightforwardly generalized to $n$ dimensions. 


# Total Variation (TV)
As the name suggests, Total variation tries to penalize variation in the image intensity. Therefore it sums up the gradient strength at each point
of the image. In 2D this is:

$\text{Reg}(S(r)) = \sum_r  |(\nabla S)(r)|$

Since we look at the magnitude of the gradient strength, this regularizer is anisotropic.

In 2D this is:

$\text{Reg}(S(r)) = \sum_{x,y} \sqrt{|S(x + 1, y) - S(x, y)|^2 + |S(x, y + 1) - S(x, y)|^2}$


# Tikhonov Regularization
The Tikhonov regularizer is not as specific defined as Good's Roughness or Total Variation. In general Tikhonov regularization is defined by:


$\text{Reg}(S(r)) = \| (\Gamma S)(r) \|_2^2$

where $\Gamma$ is an operator which can be chosen freely. Common options are the identity operator which penalizes therefore just high intensity values. Another option would be the spatial gradient which would result
in a similar operator to TV. And the last option we implemented is the spatial Laplace.


# Total Hessian Norm (TH)
The Total Hessian norm penalizes the second derivatives (curvature) of the reconstruction instead of the gradient like Total Variation does:

$\text{Reg}(S(r)) = \sum_r \sqrt{\sum_{d} \left(\frac{\partial^2 S}{\partial r_d^2}(r)\right)^2 + 2\sum_{d<e} \left(\frac{\partial^2 S}{\partial r_d r_e}(r)\right)^2}$

Smooth second order regularizers often lead to more natural reconstructions than Total Variation, which tends to introduce piecewise constant (staircase) artifacts.


# Hessian Schatten Norm (HS)
The Hessian Schatten norm considers the eigenvalues of the pixel-wise Hessian matrix $H(r) = \nabla^2 S(r)$:

$\text{Reg}(S(r)) = \sum_r \left( \sum_{l=1}^{n} |\lambda_l(r)|^p \right)^{1/p}$

where $\lambda_l(r)$ are the eigenvalues of the $2\times 2$ Hessian over the two dimensions selected by `sum_dims`. The Hessian is evaluated in closed form, so `sum_dims` must be exactly two dimensions; the remaining dimensions only take part in the summation. The pixel-wise Hessian uses centered (symmetric) second-order stencils, so the norm has no preferred direction.
