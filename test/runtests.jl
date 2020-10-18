using DeconvOptim
using Test
using FFTW, Noise, Statistics
using Random

Random.seed!(42)

@testset "Utils" begin
    include("utils.jl")
end


