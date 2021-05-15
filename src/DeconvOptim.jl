module DeconvOptim
    
export gpu_or_cpu

 # to check wether CUDA is enabled
using Requires
# for fast array regularizers
using Tullio
# optional CUDA dependency
include("requires.jl")

 # for optimization
using Optim
 #mean
using Statistics
using FFTW
FFTW.set_num_threads(4)
using LineSearches

# possible up_sampling 
using Interpolations

# for defining custom derivatives
using ChainRulesCore
using LinearAlgebra


gpu_or_cpu(x) = Array



include("forward_models.jl")
include("lossfunctions.jl")
include("mappings.jl")
# special CUDA regularizers
include("regularizer_cuda.jl")
include("regularizer.jl")
include("utils.jl")
include("generic_invert.jl")
include("lucy_richardson.jl")
include("deconvolution.jl")

# refresh Zygote to load the custom rrules defined with ChainRulesCore
using Zygote: gradient


# end module
end
