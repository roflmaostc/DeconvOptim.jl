using DeconvOptim
using Test
using FFTW, Noise, Statistics, Zygote
using Random
using TestImages, Noise
using Pkg
Pkg.add(url="https://github.com/emmt/OptimPackNextGen.jl")
using OptimPackNextGen

 # fix seed for reproducibility
Random.seed!(42)


@testset "Utils" begin
    @test gpu_or_cpu(randn((2,2))) == Array
    include("utils.jl")
end


include("conv.jl")
include("mappings.jl")

include("forward_models.jl")

include("lossfunctions.jl")

 # testing is rather hard, but include at least some basic testing
include("regularizer.jl")

include("main.jl")


