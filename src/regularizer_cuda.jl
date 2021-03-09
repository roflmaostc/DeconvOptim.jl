export TV_cuda

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
function TV_cuda(; num_dims=2)
    if num_dims == 3
        reg_TV = x -> begin
            x2 = x
            x3 = x
            x4 = x
            return sum(@tullio res[i, j, k] := sqrt(1f-8 + 
                                         abs2(x[i,j,k] - x2[i+1,j,k]) +
                                         abs2(x[i,j,k] - x3[i,j+1,k]) +
                                         abs2(x[i,j,k] - x4[i,j,k+1])))
        end
    elseif num_dims == 2
        reg_TV = x -> begin
            x2 = x
            x3 = x
            return sum(@tullio res[i, j] := sqrt(1f-8 + 
                                         abs2(x[i,j] - x2[i+1,j]) +
                                         abs2(x[i,j] - x3[i,j+1])))
        end
    else
        throw(ArgumentError("num_dims must be 2 or 3"))
    end
    
    return reg_TV
end
