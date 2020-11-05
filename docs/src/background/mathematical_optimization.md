# Mathematical Optimization
Deconvolution was already described as an optimization problem in the 1970s by [Lucy:74](@cite), [Richardson:72](@cite).
Since then, many variants and different kinds of deconvolution algorithms were presented, but mainly based on the concept of Lucy-Richardson.
We try to formulate convolution as an inverse physical problem and solve it using a convex optimization loss function so that we can use
fast optimizers to find the optimum. The variables we want to optimize for, are the pixels of the reconstruction $S(r)$. Therefore our reconstruction problem consists of several thousands to billion variables.
Mathematical the optimization can be written as:

$\underset{S(r)}{\arg \min}\, L(\text{Fwd}(S(r))) + \text{Reg}(S(r))$

where $\text{Fwd}$ represents the forward model (in our case convolution of $S(r)$ with the $\text{PSF}$), $S(r)$ is ideal reconstruction, $L$ the loss function and $\text{Reg}$ is a regularizer. The regularizer 
puts in some prior information about the structure of the object. 
See the following sections for more details about each part.

## Map Functions
In some cases we want to restrict the optimizer to solutions with $S(r) \geq 0$. Usually one uses boxed optimizer or penalties to prevent negativity.
However, in some cases, a $S(r) < 0$ can lead to issues during the optimization process. For that purpose we can introduce a mapping function.
Instead of optimizing for $S(r)$ we can optimize for some $\hat S(r)$ where $M$ is the mapping function connection 

$S(r)= M(\hat S(r)).$

A simple mapping function leading to $S(r) \geq 0$ is 

$M(\hat S(r)) = \hat S(r)^2$

The optimization problem is then given by


$\underset{\hat S(r)}{\arg \min}\, L(\text{Fwd}(M(\hat S(r)))) + \text{Reg}(M(\hat S(r)))$

After the optimization we need to apply $M$ on $\hat S$ to get the reconstructed sample 

$S(r) = M(\hat S(r))$

One could also choose different functions $M$ to obtain reconstruction in certain intensity intervals.
