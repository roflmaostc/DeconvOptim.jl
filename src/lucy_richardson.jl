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
- `progress`: if not `nothing`, the progress will be monitored in a summary dictionary as obtained by
              DeconvOptim.options_trace_deconv()

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
                                   conv_dims=1:ndims(psf),
                                   progress = nothing)


    otf, conv_temp = plan_conv(measured, psf, conv_dims) 
    otf_conj = conj.(otf)
    # initializer
    rec = abs.(conv_temp(measured, otf))#ones(eltype(measured), size(measured))
    
    # buffer for gradient
    # we need Base.invokelatest because of world age issues with generated
    # regularizers
    buffer_grad =  let 
        if !isnothing(regularizer)
            Base.invokelatest(gradient, regularizer, rec)[1]
        else
            nothing
        end
    end

    ∇reg(x) = buffer_grad .= Base.invokelatest(gradient, regularizer, x)[1]

    buffer = copy(measured)

    iter_without_reg(rec) = begin
        buffer .= measured ./ (conv_temp(rec, otf))
        conv_temp(buffer, otf_conj)
    end
    iter_with_reg(rec) = buffer .= (iter_without_reg(rec) .- λ .* ∇reg(rec))

    iter = isnothing(regularizer) ? iter_without_reg : iter_with_reg

    # the loss function is only needed for logging, not for LR itself
    loss(myrec) = begin
        fwd = conv_temp(myrec, otf)
        return sum(fwd .- measured .* log.(fwd))
    end

    # logging part
    tmp_time = 0.0
    if progress !== nothing
        record_progress!(progress, rec, 0, loss(rec), 0.0, 1.0)
        tmp_time=time()
    end
    code_time = 0.0

    # do actual optimization
    for i in 1:iterations
        rec .*= iter(rec)
        if progress !== nothing
            # do not count the time for evaluating the loss here.
            code_time += time() .- tmp_time
            record_progress!(progress, copy(rec), i, loss(rec), code_time, 1.0)
            tmp_time=time()
        end
    end

    return rec
end

