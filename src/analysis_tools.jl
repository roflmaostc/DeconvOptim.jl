export options_trace_deconv

"""
    relative_energy_regain(ground_truth, rec)

Calculates the relative energy regain between the `ground_truth`
and the reconstruction.
Assumes that both arrays are 2 dimensional

# Reference
* [Rainer Heintzmann, \"Estimating missing information by maximum likelihood deconvolution\"](https://www.sciencedirect.com/science/article/abs/pii/S0968432806001272)
"""
function relative_energy_regain(ground_truth, rec)
    T = eltype(ground_truth)
    # go to fourier space
    ground_truth_fft = fft(ground_truth)
    rec_fft = fft(rec)

    # a dict to store the values for certain frequencies
    # we store a list since some (rounded) frequencies occur more than once
    ΔE_R_dict = Dict{T, Vector{T}}()
    E_R_dict = Dict{T, Vector{T}}()

    # round the frequencies to 4 digits, alternative would be to bin
    round4(x) = T(round(x, digits=3))
    
    
    # iterate over the frequencies and calculate the relative energy regain
    for (i₂, f₂) in enumerate(fftfreq(size(rec_fft, 2)))
        for (i₁, f₁) in enumerate(fftfreq(size(rec_fft, 1)))
            f_res = round4(√(f₁^2 + f₂^2))
            Δ_E_R = abs2(ground_truth_fft[i₁, i₂] - rec_fft[i₁, i₂]) 
            E_R = abs2(ground_truth_fft[i₁, i₂]) 

            update_dict_list!(ΔE_R_dict, f_res, Δ_E_R)
            update_dict_list!(E_R_dict, f_res, E_R)
        end
    end
    
    
    # finally transform everything into a list of frequencies and 
    # a list of relative energy regains
    freqs = T[]
    G_R_list = T[]
    for f in sort(T.(keys(ΔE_R_dict)))
        push!(freqs, f)
        mean_ΔE_r = mean(ΔE_R_dict[f])
        mean_E_r = mean(E_R_dict[f])
        push!(G_R_list, (mean_E_r - mean_ΔE_r) / mean_E_r)
    end
    
    return freqs, G_R_list
end




"""
    update_dict_list!(d, k, v)

Updates the dict `d` which stores a list.
If `k` is in the keys of `d` we simply push `v` to the list
otherwise create a new list `[v]`
"""
function update_dict_list!(d, k, v)
    if haskey(d, k)
        push!(d[k], v)
    else
        d[k] = [v]
    end
    return d
end


"""
    normalized_cross_correlation(ground_truth, measured)

Calculates the normalized cross correlation.

External links: 
* [Wikipedia](https://en.wikipedia.org/wiki/Sombrero_function)
* [StatsBase.jl](https://juliastats.org/StatsBase.jl/stable/signalcorr/#StatsBase.crosscor)
"""
function normalized_cross_correlation(ground_truth, measured)
    fl(x) = collect(Iterators.flatten(x))
    ground_truth = fl(ground_truth)
    measured = fl(measured)

    ncc = crosscor(ground_truth, measured, [0], demean=true)[begin]
    return ncc
end

"""
    normalized_variance(a, b)

Calculates the mean variance between two array, but normalizing arra a to the same mean as array b.
"""
function normalized_variance(a, b)
    factor = sum(b)/sum(a)
    sum(abs2.(a.*factor .-b))./prod(size(a))
end

function reset_summary!(summary)
    summary["losses"] = []
    summary["best_ncc"] = -Inf
    summary["best_ncc_idx"] = 0
    summary["best_ncc_img"] = []
    summary["nccs"] = []
    summary["best_nvar"] = Inf
    summary["best_nvar_idx"] = 0
    summary["best_nvar_img"] = []
    summary["times"] = []
    summary["step_sizes"] = []
    summary["nvars"] = []
end

"""
    options_trace_deconv(ground_truth, iterations, mapping, every=1)

    A useful routine to simplify performance checks of deconvolution on simulated data.
    Returns an Options structure to be used with the deconvolution routine as an argument to `opt_options` and 
    a summary dictionary with all the performance metrics calculated, which is resetted and updated during deconvolution.
    This can then be plotted or visualized.
    The summary dictionary has the following content:

    "best_nvar"     => the lowest normalized variance compared to the `ground_truth` that was achieved.
    "best_nvar_img" => the reconstruction result corresponding to this lowest normalized variance
    "best_nvar_idx" => the corresponding index where this was achieved. `(best_nvar_idx-1)*every+1` approximated the iteration number.
    "best_ncc"      => the highest normalized crosscorrelation compared to the `ground_truth` that was achieved.
    "best_ncc_img"  => the reconstruction result corresponding to this highest normalized crosscorrelation
    "losses"        => the vector of losses evaluated at each of `every` iterations.
    "nccs"          => the vector of normalized cross correlations calculated at each of `every` iterations.
    "best_ncc_idx"  => the corresponding index where this was achieved. `(best_ncc_idx-1)*every+1` approximated the iteration number.
    "nvars"         => the vector of normalized variances calculated at each of `every` iterations.

    For an example of how to plot the results, see the file `` in the `examples` folder.
# Arguments
- `ground_truth`: The underlying ground truth data. Note that this scaling is unimportant due to the normalized norms used for comparison, 
                  whereas the relative offset matters. 
- `iterations`: The maximal number of iterations to performance. If covergence is reached, the result may have less iterations
- `mapping`: If mappings such as the positivity contraints (e.g. `NonNegative()`) are used in the deconvolution routing, they also 
             need to be provided here. Otherwises select `nothing`.
- `every`: This option allows to select every how many iterations the evaluation is performed. Note that the results will not keep track 
        of this iteration number. 

# Example
```julia-repl
julia> using DeconvOptim, TestImages, Noise, Plots;

julia> obj = Float32.(testimage("resolution_test_512"));

julia> psf = Float32.(generate_psf(size(obj), 30));

julia> img_b = conv(obj, psf);

julia> img_n = poisson(img_b, 300);

julia> iterations = 100;

julia> opt_noreg, show_noreg = options_trace_deconv(obj, iterations, Non_negative());

julia> res_noreg, o = deconvolution(img_n, psf, regularizer = nothing, opt_options=opt_noreg);

julia> opt_GR, show_GR = options_trace_deconv(obj, iterations, Non_negative());

julia> res_GR, o = deconvolution(img_n, psf, λ=1e-2, regularizer=DeconvOptim.GR(), opt_options=opt_GR);

julia> opt_TV, show_TV = options_trace_deconv(obj, iterations, Non_negative());

julia> res_TV, o = deconvolution(img_n, psf, λ=1e-3, regularizer=DeconvOptim.TV(), opt_options=opt_TV);

julia> plot()

julia> show_noreg(false,"NoReg")

julia> show_GR(false,"GR")

julia> show_TV(false,"TV")

julia> using View5D

julia> @vt (ground_truth, best_ncc_img, best_nvar_img) = show_noreg(true)
```
"""
function options_trace_deconv(ground_truth, iterations, mapping, every=1)
    summary = Dict()
    reset_summary!(summary)
    idx = 1
    cb = tr -> begin
        # iteration always starts with index 0 (before 1st iteration)
        if tr[end].iteration == 0
            reset_summary!(summary)
            idx = 1
        end
        loss = tr[end].value
        push!(summary["losses"], loss)
        push!(summary["times"], tr[end].metadata["time"])
        push!(summary["step_sizes"], tr[end].metadata["Current step size"])
        # current image:
        img = (mapping === nothing) ? tr[end].metadata["x"] : mapping[1](tr[end].metadata["x"]) 
        # the line below is needed, since in the iterations, the measurement is rescaled to a mean of one.
        # see deconvolution.jl.  This rescaling is only an estimate and does not affect the norms.
        img *= mean(ground_truth)

        ncc = DeconvOptim.normalized_cross_correlation(ground_truth, img)
        push!(summary["nccs"], ncc)
        summary["best_ncc"], summary["best_ncc_img"], summary["best_ncc_idx"] = let
            if ncc > summary["best_ncc"]
                    (ncc, img, idx) 
            else
                    (summary["best_ncc"], summary["best_ncc_img"], summary["best_ncc_idx"])
            end
        end
        nvar = normalized_variance(img, ground_truth)
        push!(summary["nvars"], nvar)
        summary["best_nvar"], summary["best_nvar_img"], summary["best_nvar_idx"] = let 
            if nvar < summary["best_nvar"]
                    (nvar, img, idx)
            else
                    (summary["best_nvar"], summary["best_nvar_img"], summary["best_nvar_idx"])
            end
        end
        idx += 1
        false
    end

    opt_options = Optim.Options(callback = cb, iterations=iterations, show_every=every, store_trace=true, extended_trace=true)
    return (opt_options, summary)
end
