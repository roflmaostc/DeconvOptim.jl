{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "using Pkg, Revise\n",
    "Pkg.activate(\"paper_figures/.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 254,
   "metadata": {},
   "outputs": [],
   "source": [
    "using Tullio, BenchmarkTools, CUDA, CUDAKernels, KernelAbstractions, Zygote, Distributed\n",
    "CUDA.allowscalar(false)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 367,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "TV_tullio_2 (generic function with 2 methods)"
      ]
     },
     "execution_count": 367,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "function TV_cpu(arr, ϵ=eltype(arr)(1e-8))\n",
    "    @tullio r = sqrt(ϵ + abs2(arr[i,j,k] - arr[i+1,j,k])\n",
    "                       + abs2(arr[i,j,k] - arr[i,j+1,k])\n",
    "                       + abs2(arr[i,j,k] - arr[i,j,k+1]))\n",
    "end\n",
    "\n",
    "# see here https://github.com/mcabbott/Tullio.jl/issues/85\n",
    "# gradient pass is super slow\n",
    "function TV_gpu(arr, ϵ=eltype(arr)(1e-8))\n",
    "    arr1 = arr\n",
    "    arr2 = arr\n",
    "    arr3 = arr\n",
    "    @tullio r[i, j, k] := sqrt(ϵ + abs2(arr[i,j,k] - arr1[i+1,j,k])\n",
    "                       + abs2(arr[i,j,k] - arr2[i,j+1,k])\n",
    "                       + abs2(arr[i,j,k] - arr3[i,j,k+1]))\n",
    "    return sum(r)\n",
    "end\n",
    "\n",
    "\n",
    "f_inds(rs, b) = ntuple(i -> i == b ? rs[i] .+ 1 : rs[i], length(rs))\n",
    "\n",
    "# called \"naiv\" implementation\n",
    "function TV_3D_view(arr::AbstractArray{T, N}, ϵ=1f-8) where {T, N}\n",
    "    as = ntuple(i -> axes(arr, i), Val(N))\n",
    "    rs = map(x -> first(x):last(x)-1, as)\n",
    "    arr0 = view(arr, f_inds(rs, 0)...)\n",
    "    arr1 = view(arr, f_inds(rs, 1)...)\n",
    "    arr2 = view(arr, f_inds(rs, 2)...)\n",
    "    arr3 = view(arr, f_inds(rs, 3)...)\n",
    "\n",
    "    return @fastmath sum(sqrt.(ϵ .+ abs2.(arr1 .- arr0) .+ \n",
    "                            abs2.(arr2 .- arr0) .+ abs2.(arr3 .- arr0)))\n",
    "\n",
    "end\n",
    "\n",
    "function TV_3D_circshift(arr::AbstractArray{T, N}, ϵ=1f-8) where {T, N}\n",
    "    arr0 = arr\n",
    "    arr1 = circshift(arr, (1, 0, 0))\n",
    "    arr2 = circshift(arr, (0, 1, 0))\n",
    "    arr3 = circshift(arr, (0, 0, 1))\n",
    "\n",
    "    return @fastmath sum(sqrt.(ϵ .+ abs2.(arr1 .- arr0) .+ \n",
    "                            abs2.(arr2 .- arr0) .+ abs2.(arr3 .- arr0)))\n",
    "\n",
    "end\n",
    "\n",
    "function TV_tullio_2(arr::AbstractArray{T, N}, ϵ=1f-8) where {T, N}\n",
    "    @tullio diff1[i,j,k] := abs2(arr[i+1,j,k] - arr[i,j,k])\n",
    "    @tullio diff2[i,j,k] := abs2(arr[i,j+1,k] - arr[i,j,k])\n",
    "    @tullio diff3[i,j,k] := abs2(arr[i,j,k+1] - arr[i,j,k])\n",
    "\n",
    "    @tullio res[i, j, k] := sqrt(ϵ + diff1[i+0,j+0,k+0] + diff2[i+0,j+0,k+0] + diff3[i+0,j+0,k+0]) (i in 1:size(arr, 1)-1, j in 1:size(arr, 2)-1,l in 1:size(arr, 3)-1)\n",
    "    return sum(res)\n",
    "end"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 184,
   "metadata": {},
   "outputs": [],
   "source": [
    "arr = randn(Float32, (300, 300, 300));\n",
    "arr_c = CuArray(arr);"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 185,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  5.528 ms (217 allocations: 10.52 KiB)\n",
      "  119.764 ms (312 allocations: 103.01 MiB)\n"
     ]
    }
   ],
   "source": [
    "@btime TV_cpu(arr)\n",
    "x = @btime gradient(TV_cpu, arr);"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 186,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  83.876 ms (3 allocations: 101.97 MiB)\n",
      "  7.177 s (320772165 allocations: 8.87 GiB)\n"
     ]
    }
   ],
   "source": [
    "@btime TV_3D_view(arr);\n",
    "x = @btime gradient(TV_3D_view, arr);"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 187,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  2.497 ms (13435 allocations: 239.41 KiB)\n",
      "  22.900 ms (141460 allocations: 2.25 MiB)\n"
     ]
    }
   ],
   "source": [
    "@btime TV_3D_view($arr_c);\n",
    "x_c = @btime gradient(TV_3D_view, $arr_c);"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 381,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  2.524 ms (6698 allocations: 134.14 KiB)\n",
      "  35.236 ms (133844 allocations: 2.13 MiB)\n"
     ]
    }
   ],
   "source": [
    "GC.gc(true)\n",
    "@btime CUDA.@sync TV_3D_view($arr_c);\n",
    "GC.gc(true)\n",
    "@btime CUDA.@sync gradient(TV_3D_view, $arr_c);"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 380,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  5.400 ms (8215 allocations: 148.30 KiB)\n",
      "  36.265 ms (184331 allocations: 2.84 MiB)\n"
     ]
    }
   ],
   "source": [
    "GC.gc(true)\n",
    "@btime CUDA.@sync TV_3D_circshift($arr_c);\n",
    "GC.gc(true)\n",
    "@btime CUDA.@sync gradient(TV_3D_circshift, $arr_c);"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 382,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  0.312808 seconds (605.20 k allocations: 133.668 MiB, 87.30% compilation time)\n",
      "  0.843808 seconds (2.06 M allocations: 720.621 MiB, 3.83% gc time, 58.58% compilation time)\n"
     ]
    }
   ],
   "source": [
    "@time TV_gpu(arr)\n",
    "@time gradient(TV_gpu, arr);"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Julia 1.6.1",
   "language": "julia",
   "name": "julia-1.6"
  },
  "language_info": {
   "file_extension": ".jl",
   "mimetype": "application/julia",
   "name": "julia",
   "version": "1.6.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
