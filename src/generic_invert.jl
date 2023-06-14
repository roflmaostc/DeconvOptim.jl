export invert, OptInterface, Opt_Optim, Opt_OptimPackNextGen

abstract type OptInterface end  # To accommodate multiple optimizers which are incompatible
struct Opt_Optim <: OptInterface end  # 
struct Opt_OptimPackNextGen <: OptInterface end  # 

"""
    invert(measured, rec0, forward; <keyword arguments>)

Tries to invert the `forward` model. `forward` is a function taking 
an input with the shape of `rec0` and returns an object which has the 
same shape as `measured`

Multiple keyword arguments can be specified for different loss functions,
regularizers and mappings.

# Arguments
- `loss=Poisson()`: the loss function being compatible to compare with `measured`.
- `regularizer=nothing`: A regularizer function, same form as `loss`.
- `λ=0.05`: A float indicating the total weighting of the regularizer with 
    respect to the global loss function
- `mapping=Non_negative()`: Applies a mapping of the optimizer weight. Default is a 
              parabola which achieves a non-negativity constraint.
- `iterations=nothing`: Specifies a number of iterations after the optimization.
    definitely should stop. Will be overwritten if `opt_options` is provided. Default: 20
- `opt_package=Opt_Optim`: decides which backend for the optimizer is used.
- `opt=LBFGS()`: The chosen optimizer which must fit to `opt_package`. 
- `opt_options=nothing`: Can be a options file required by Optim.jl. Will overwrite iterations.
- `debug_f=nothing`: A debug function which must take a single argument, the current reconstruction. 
"""
function invert(measured, rec0, forward;
    iterations=nothing, λ=eltype(rec0)(0.05),
    regularizer=nothing,
    opt=LBFGS(linesearch=LineSearches.BackTracking()),
    opt_options=nothing,
    mapping=Non_negative(),
    loss=Poisson(),
    real_gradient=true,
    debug_f=nothing,
    opt_package=Opt_Optim)

    # if not special options are given, just restrict iterations
    if opt_package <: Opt_Optim && opt_options !== nothing && iterations !== nothing
        error("If `opt_options` are provided you need to include the iterations as part of these instead of providing the `iterations` argument.")
    end
    iterations = (iterations === nothing) ? 20 : iterations

    if opt_package <: Opt_Optim
        if opt_options === nothing
            opt_options = Optim.Options(iterations=iterations)
        end
    end

    # Get the mapping functions to achieve constraints
    # like non negativity
    mf, m_invf = get_mapping(mapping)
    regularizer = get_regularizer(regularizer, eltype(rec0))


    debug_f_n(x) =
        let
            if isnothing(debug_f)
                identity(x)
            else
                debug_f(mf(x))
            end
        end

    storage_μ = deepcopy(measured)
    function total_loss(rec)
        # handle if there is a provided mapping function
        mf_rec = mf(rec)
        forward_v = forward(mf_rec)
        loss_v = sum(loss(forward_v, measured, storage_μ))
        loss_v += λ .* regularizer(mf_rec)
        return loss_v
    end
    # nice precompilation before calling Zygote etc.
    Base.invokelatest(total_loss, rec0)

    # see here https://github.com/FluxML/Zygote.jl/issues/342
    take_real_or_not(g) = real_gradient ? real.(g) : g
    take_real_or_not(g::AbstractArray{<:Real}) = g

    # this is the function which will be provided to Optimize
    # check Optim's documentation for the purpose of F and Get
    # but simply speaking F is the loss value and G it's gradient
    # depending whether one of them is nothing, we skip some computations
    # we need to call Base.invokelatest because the regularizer is a function
    # generated at runtime with eval.
    # This leads to the common "world age problem" in Julia
    # for more details on that check:
    # https://discourse.julialang.org/t/dynamically-create-a-function-initial-idea-with-eval-failed-due-to-world-age-issue/49139/17
    function fg!(F, G, rec)

        # Zygote calculates both derivative and loss, therefore do everything in one step
        if G !== nothing
            # apply debug function
            debug_f_n(rec)

            y, back = Base.invokelatest(Zygote._pullback, total_loss, rec)
            # calculate gradient
            G .= take_real_or_not(Base.invokelatest(back, 1)[2])
            if F !== nothing
                return y
            end
        end
        if F !== nothing
            return Base.invokelatest(total_loss, rec)
        end
    end

    if isa(opt_package, Type{Opt_Optim})
        if opt_options === nothing
            opt_options = Optim.Options(iterations=iterations)
        end
        # do the optimization with LBGFS
        res = Optim.optimize(Optim.only_fg!(fg!), rec0, opt, opt_options)
        res_out = mf(Optim.minimizer(res))
        # supports a different interface as for example used in OptimPackNextGen for the function 'vmlmb!'
    elseif isa(opt_package, Type{Opt_OptimPackNextGen})
        res = copy(rec0)

        if isnothing(opt_options)
            opt((x, g) -> fg!(true, g, x), res; maxiter=iterations)
        else
            opt((x, g) -> fg!(true, g, x), res; maxiter=iterations, opt_options...)
        end
        res_out = mf(res)
    else
        error("Unknown optimizer interface $(typeof(opt_package))")
    end


    return res_out, res
end
