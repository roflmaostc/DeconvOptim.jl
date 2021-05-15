export richardson_lucy_iterative

"""
    richardson_lucy_iterative(measured, psf; <keyword arguments>)

Classical iterative Richardson-Lucy iteration scheme for deconvolution.
`measured` is the measured array and `psf` the point spread function.
Converges slower than the optimization approach of `deconvolution`

# Keyword Arguments
- `regularizer=GR()`: A regularizer function. Can be exchanged
- `λ=0.05`: A float indicating the total weighting of the regularizer with 
    respect to the global loss function
- `iterations=100`: Specifies number of iterations.


# Example
```julia-repl
julia> using DeconvOptim, TestImages, Colors, Noise;

julia> img = Float32.(testimage("resolution_test_512"));

julia> psf = Float32.(generate_psf(size(img), 30));

julia> img_b = conv_psf(img, psf);

julia> img_n = poisson(img_b, 300);

julia> @time res = richardson_lucy_iterative(img_n, psf);
```
"""
function richardson_lucy_iterative(measured, psf; 
                                   regularizer=GR(),
                                   λ=0.05,
                                   iterations=100,
                                   conv_dims=1:ndims(psf))

    otf, conv = plan_conv_r(psf, measured, conv_dims) 
    otf_conj = conj.(otf)
   
    ∇reg(x) = gradient(regularizer, x)[1]

    iter_without_reg(rec) = (conv(measured ./ (conv(rec, otf)), otf_conj))
    iter_with_reg(rec) =  (iter_without_reg(rec) .- λ .* Base.invokelatest(∇reg, rec))

    iter = isnothing(regularizer) ? iter_without_reg : iter_with_reg

    rec = conv_otf_r(measured, otf) 
    for i in 1:iterations
        rec .*= iter(rec)
    end

    return rec
end

