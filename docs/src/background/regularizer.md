# Regularizer

Regularizer are commonly used in inverse problems and especially in deconvolution to obtain solutions which are optimal with respect to some prior. 
So far we have included three common regularizer. The regularizer take the current reconstruction $S(r)$ as argument and return a scalar value. This value should be also minimized and is also added
to the loss function.
Each regularizer produces some characteristic image styles.


# Good's Roughness (GR)
The Good's roughness definition was taken from [Good:71](@cite) and [Verveer:98](@cite).
For Good's roughness several identical expressions can be derived. We implemented the following one:

$\text{Reg}(S(r)) = \sum_r \sqrt{S(r)} (\Delta_N \sqrt{S})(r)$

where $N$ is the dimension of $S(r)$. $\sqrt S$ is applied elementwise.
$\Delta_d S(r)$ is the n-dimensional discrete Laplace operator. As 2D example where $r = (x,y)$:

$(\Delta_N \sqrt{S})(r) = \frac{\sqrt{S(x + s_x, y)} + \sqrt{S(x - s_x, y)} + \sqrt{S(x, y+s_y)} + \sqrt{S(x, y-s_y)} - 4 \cdot \sqrt{S(x, y)}}{s_x \cdot s_y}$

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
