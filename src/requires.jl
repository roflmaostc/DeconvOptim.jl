
isgpu(x) = false

function __init__()
    @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" @eval using KernelAbstractions
   
    @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" begin
        @info "DeconvOptim.jl: CUDA.jl is loaded, so include GPU functionality. Additionally load KernelAbstractions.jl"
        # need to import it within this package so that Tullio can generate together with KernelAbstractions.jl
        # fast GPU regularizers
        using .CUDA 
        
        gpu_or_cpu(x) = CUDA.CuArray
        isgpu(x::CUDA.CuArray) = true
        # prevent slow scalar indexing on GPU
        CUDA.allowscalar(false);

        # we need to fix some operations so that they are fast o GPUs
        # Reference: https://discourse.julialang.org/t/cuarray-and-optim/14053
        LinearAlgebra.norm1(x::CUDA.CuArray{T,N}) where {T,N} = sum(abs, x); # specializes the one-norm
        LinearAlgebra.normInf(x::CUDA.CuArray{T,N}) where {T,N} = maximum(abs, x); # specializes the one-norm
        Optim.maxdiff(x::CUDA.CuArray{T,N},y::CUDA.CuArray{T,N}) where {T,N} = maximum(abs.(x-y));

        # special CUDA regularizers
        include("regularizer_cuda.jl")
    
    end


end
