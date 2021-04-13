export invert 


function invert(measured, rec0, forward; 
                iterations=10, λ=0.05,
                regularizer=TV(num_dims=3, sum_dims=3),
                optim_optimizer=LBFGS(linesearch=LineSearches.BackTracking()),
                optim_options=nothing,
                mapping=Non_negative(),
                loss=Poisson())

    # if not special options are given, just restrict iterations
    if optim_options == nothing
        optim_options = Optim.Options(iterations=iterations)
    end


    λ = eltype(rec0)(λ)
    
    # Get the mapping functions to achieve constraints
    # like non negativity
    mf, m_invf = get_mapping(mapping)
    regularizer = get_regularizer(regularizer, eltype(rec0))

    function total_loss(rec)
        # handle if there is a provided mapping function
        mf_rec = mf(rec) 
        forward_v = forward(mf_rec)
        loss_v = sum(loss(forward_v, measured))
        loss_v += λ .* regularizer(mf_rec) 
        return loss_v 
    end
    # nice precompilation before calling Zygote etc.
    Base.invokelatest(total_loss, rec0)

    # this is the function which will be provided to Optimize
    # check Optim's documentation for the purpose of F and Get
    # but simply speaking F is the loss value and G it's gradient
    # depending whether one of them is nothing, we skip some computations
    # we need to call Base.invokelatest because the regularizer is a function
    # generated at runtime with eval.
    # This leads to the common "world age problem" in Julia
    # for more details on that check:
    # https://discourse.julialang.org/t/dynamically-create-a-function-initial-idea-with-eval-failed-due-to-world-age-issue/49139/17
    function f!(F, G, rec)
        # Zygote calculates both derivative and loss, therefore do everything in one step
        if G != nothing
            y, back = Base.invokelatest(Zygote._pullback, total_loss, rec)
            # calculate gradient
            G .= Base.invokelatest(back, 1)[2]
            if F != nothing
                return y
            end
        end
        if F != nothing
            return Base.invokelatest(total_loss, rec)
        end
    end

    optim_options = Optim.Options(iterations=iterations)
    
    # do the optimization with LBGFS
    res = Optim.optimize(Optim.only_fg!(f!), rec0, optim_optimizer, optim_options)
    res_out = mf(Optim.minimizer(res))


    return res_out, res
end
