export TV_cuda

f_inds(rs, b) = ntuple(i -> i == b ? rs[i] .+ 1 : rs[i], length(rs))


"""
    TV_cuda(; num_dims=2)
This function returns a function to calculate the Total Variation regularizer 
of a 2 or 3 dimensional array.
`num_dims` can be either `2` or `3`.

```julia-repl
julia> using CUDA

julia> reg = TV_cuda(num_dims=2);

julia> reg(CuArray([1 2 3; 4 5 6; 7 8 9]))
12.649111f0
```
"""
function TV_cuda(; num_dims=2, weights=ones(Float32, num_dims), ϵ=1f-8)
    if num_dims == 3
        return arr -> TV_3D_view(arr, weights, ϵ)
    elseif num_dims == 2
        return arr -> TV_2D_view(arr, weights, ϵ)
    else
        throw(ArgumentError("num_dims must be 2 or 3"))
    end
    
    return reg_TV
end

function TV_2D_view(arr::AbstractArray{T, N}, weights, ϵ=1f-8) where {T, N}
    as = ntuple(i -> axes(arr, i), Val(N))
    rs = map(x -> first(x):last(x)-1, as)
    arr0 = view(arr, f_inds(rs, 0)...)
    arr1 = view(arr, f_inds(rs, 1)...)
    arr2 = view(arr, f_inds(rs, 2)...)
    
    return @fastmath sum(sqrt.(ϵ .+ weights[1] .* (arr1 .- arr0).^2 .+ weights[2] .* (arr0 .- arr2).^2))
end

function TV_3D_view(arr::AbstractArray{T, N}, weights, ϵ=1f-8) where {T, N}
    as = ntuple(i -> axes(arr, i), Val(N))
    rs = map(x -> first(x):last(x)-1, as)
    arr0 = view(arr, f_inds(rs, 0)...)
    arr1 = view(arr, f_inds(rs, 1)...)
    arr2 = view(arr, f_inds(rs, 2)...)
    arr3 = view(arr, f_inds(rs, 3)...)
    
    return @fastmath sum(sqrt.(ϵ .+ weights[1] .* (arr1 .- arr0).^2 .+ 
                               weights[2] .* (arr2 .- arr1).^2 .+  weights[3] .* (arr3 .- arr0).^2 ))
end
