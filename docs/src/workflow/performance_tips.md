# Performance Tips

## Regularizer
The regularizers are built during calling with metaprogramming. Everytime you call `TV()` it creates a new version which is evaluated with
`eval`. In the first `deconvolution` routine it has to compile this piece of code.
To prevent the compilation every time, define 
```julia
reg = TV()
```
which is then later used as a variable. In a notebook or REPL environment just define it in a different cell.

## No Regularizer
Often the results are good without regularizer but then need to be early stopped (e.g. like `iterations=20`).
This increases the performance drastically, but might lead to more artifacts in certain regions.


## Optimizer
### L-BFGS
You can also try to adjust the settings of the [L-BFGS algorithm](https://julianlsolvers.github.io/Optim.jl/stable/#algo/lbfgs/) of Optim.jl
Try to change `m` in `opt=LBFGS(linesearch=BackTracking(), m=10)`.
`m` is the history value of the L-BFGS algorithm. Smaller is usually faster, but might lead to worse results. 
See also [Wikipedia](https://en.wikipedia.org/wiki/Limited-memory_BFGS).


### Line Search
L-BFGS uses [LineSearches.jl](https://github.com/JuliaNLSolvers/LineSearches.jl). In our examples `BackTracking` turned
out to be the fastest, but it might be worth to try different ones.

### Iterations
Try to set the keyword `iterations=20` to a lower number if you want to early stop the deconvolution.
Of course, the results might be worse then.
