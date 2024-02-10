using DeconvOptim
using Test
using FFTW, Noise, Statistics, Zygote
using Random
using TestImages, Noise
using Pkg
#Pkg.add(url="https://github.com/emmt/OptimPackNextGen.jl")
#using OptimPackNextGen

 # fix seed for reproducibility
Random.seed!(42)


@testset "Utils" begin
    include("utils.jl")
end


include("analysis_tools.jl")
include("hessian_schatten_norm.jl")
include("conv.jl")
include("mappings.jl")

include("forward_models.jl")

include("lossfunctions.jl")

 # testing is rather hard, but include at least some basic testing
include("regularizer.jl")

include("main.jl")


