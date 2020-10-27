# Loss functions
Loss functions are generally introduced in mathematical optimization theory.
The purpose is to map a certain optimization problem onto a real number.
By minimising this real number, one hopes that the obtained parameters provide
a useful result for the problem. 
One common loss function (especially in deep learning) is simply the $L^2$ norm between measurement and prediction.


So far we provide two adapted loss functions with our package. However, it is relatively easy to incorporate
custom defined loss functions or import them from packages like [Flux.ml](https://fluxml.ai/Flux.jl/stable/models/losses/).
The interface from Flux.ml is the same as for our loss functions.


## Poisson Loss
As mentioned in [Noise Model](@ref), Poisson shot noise is usually the dominant source of noise.
Therefore one achieves good results by choosing a loss function which considers both the difference between measurement and reconstruction but also the noise process.
See [Verveer:98](@cite) and [Mertz:2019](@cite) for more details on that.
As key idea we interpret the measurement as a stochastic process. Our aim is to find a deconvolved image which describes as accurate as possible the measured image.
Mathematically the probability for a certain measurement $Y$ is

$p(Y(r)|\mu(r)) = \prod_r \frac{\mu(r)^{Y(r)}}{\Gamma(Y(r) + 1)} \exp(- \mu(r))$

where $Y$ is the measurement, $\mu$ is the expected measurement (ideal measurement without noise) and $\Gamma$ is the generalized factorial function.
In the deconvolution process we get $Y$ as input and want to find the ideal specimen $S$ which results in a measurement $\mu(r) = (S * \text{PSF})(r))$.
Since we want to find the best reconstruction, we want to find a $\mu(r)$ so that $p(Y(r) | \mu(r))$ gets as large as possible. Because that means
that we find the specimen which describes the measurement with the highest probability.
Instead of maximising $p(Y(r) | \mu(r))$ a common trick is to minimise $- \log(p(Y(r)|\mu(r)))$. 
Mathematically, the optimization of both functions provides same results but the latter is numerically more stable.

$ \underset{S(r)}{\arg \min} (- \log(p(Y(r)|\mu(r)))) = \underset{S(r)}{\arg \min} \sum_r (\mu(r) + \log(\Gamma(Y(r) + 1)) - Y(r) \log(\mu(r))$ 

which is equivalent to

$\underset{S(r)}{\arg \min} L = \underset{S(r)}{\arg \min} \sum_r (\mu(r)  - Y(r) \log(\mu(r))$

since the second term only depends on $Y(r)$ but not on $\mu(r)$.
The gradient of $L$ with respect to $\mu(r)$ is simply

$\nabla L = 1 - \frac{Y(r)}{\mu(r)}.$

The function $L$ and the gradient $\nabla L$ are needed for any gradient descent optimization algorithm.

