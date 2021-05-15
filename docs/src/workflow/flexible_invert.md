# Flexible Invert
We also provide functionality to invert problems which are not a straightforward deconvolution
like multi view convolution or a problem where several measurements with different properties and forward models are available. 
The idea is that a `forward` model, a initial guess and the according measurements are in principle enough to invert the problem.

## Example
Look into the [examples folder](https://github.com/roflmaostc/DeconvOptim.jl/tree/master/examples/generic_invert.ipynb) to see how it can work.
