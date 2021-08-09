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

julia> img_b = conv(img, psf);

julia> img_n = poisson(img_b, 300);

julia> @time res = richardson_lucy_iterative(img_n, psf);
```
"""
function richardson_lucy_iterative(measured, psf; 
                                   regularizer=GR(),
                                   λ=0.05,
                                   iterations=100,
                                   conv_dims=1:ndims(psf))


    otf, conv_temp = plan_conv(measured, psf, conv_dims) 
    otf_conj = conj.(otf)
    # initializer
    rec = abs.(conv_temp(measured, otf))#ones(eltype(measured), size(measured))
    
    # buffer for gradient

    buffer_grad =  let 
        if !isnothing(regularizer)
            gradient(regularizer, rec)[1]
        else
            nothing
        end
    end

    ∇reg(x) = buffer_grad .= gradient(regularizer, x)[1]

    buffer = copy(measured)

    iter_without_reg(rec) = begin
        buffer .= measured ./ (conv_temp(rec, otf))
        conv_temp(buffer, otf_conj)
    end
    iter_with_reg(rec) = buffer .= (iter_without_reg(rec) .- λ .* Base.invokelatest(∇reg, rec))

    iter = isnothing(regularizer) ? iter_without_reg : iter_with_reg

    for i in 1:iterations
        rec .*= iter(rec)
    end


    return rec
end

