# here we compare various deconvolution options in terms of image quality
using IndexFunArrays, LinearAlgebra, Random, Noise, TestImages, FourierTools
using DeconvOptim, FFTW, Optim, LineSearches
using View5D, Plots

obj = 100f0 .* Float32.(testimage("resolution_test_512"))
# simulate a simple PSF
sz = size(obj); 
R_max = sz[1] ./ 12.0;
psf = generate_psf(sz, R_max);

# simulate a perfect image
conv_img = DeconvOptim.conv(obj, psf);

# set a fixed point for the measured data quality
max_photons = 1000

Random.seed!(42)
measured = poisson(conv_img, max_photons);

opt_options = nothing

iterations = 100
function get_data(summary)
    return (summary["best_ncc_img"], summary["best_nvar_img"])
end
function show_ncc!(summary, title="")
        nccs = summary["nccs"]
        plt = plot!(nccs, label=title*" NCC")
        col = plt[1][end].plotattributes[:markercolor]
        vline!([summary["best_ncc_idx"]], line=:dash, color=col, label=title*"_best NCC")
        xlabel!("iteration")
        ylabel!("normalized cross correlation")
    end

function show_nvar!(summary, title="")
    nvars = summary["nvars"]
    nvars_norm = nvars ./ nvars[1]
    plt = plot!(nvars_norm, label=title*" NVAR")
    col = plt[1][end].plotattributes[:markercolor]
    vline!([summary["best_nvar_idx"]], line=:dash, color=col, label=title*"_best NVAR")
    xlabel!("iteration")
    ylabel!("normalized variance")
end

function show_loss!(summary, addCurves="", lowest_loss=minimum(summary["losses"]))
        losses = summary["losses"]
        log_losses = log.(losses .- lowest_loss .+ 1)
        rel_losses = log_losses # .- maximum(log_losses)
        plot!(rel_losses[1:end-1], label=addCurves)
        xlabel!("iteration")
        ylabel!("log loss")
end

opt_options, noreg_summary = DeconvOptim.options_trace_deconv(obj, iterations, Non_negative());

res_noreg = deconvolution(measured, psf;
        regularizer=nothing, mapping=Non_negative(),
        opt_options=opt_options, debug_f=nothing)

opt_grad, grad_summary = DeconvOptim.options_trace_deconv(obj, iterations, Non_negative(), );
res_grad = deconvolution(measured, psf;
        regularizer=nothing,  mapping=Non_negative(),
        opt=GradientDescent(), opt_options=opt_grad, debug_f=nothing)
        
opt_gr, gr_summary = DeconvOptim.options_trace_deconv(obj, iterations, Non_negative());
res_gr = deconvolution(measured, psf;
        regularizer=DeconvOptim.GR(), λ=1e-3, 
        mapping=Non_negative(), opt_options=opt_gr, debug_f=nothing)

opt_tv, tv_summary = DeconvOptim.options_trace_deconv(obj, iterations, Non_negative())
res_tv = deconvolution(measured, psf;
        regularizer=DeconvOptim.TV(), λ=1e-3, 
        mapping=Non_negative(), opt_options=opt_tv, debug_f=nothing)

plot()
title!("Regularization")
do_display=false
show_ncc!(noreg_summary, "NoReg")
show_nvar!(noreg_summary, "NoReg")
show_ncc!(gr_summary, "GR")
show_nvar!(gr_summary, "GR")
show_ncc!(tv_summary, "TV")
show_nvar!(tv_summary, "TV")

plot()
title!("Optimization")
show_loss!(noreg_summary, "LBFGS", minimum(noreg_summary["losses"]))
show_loss!(grad_summary, "SteepestDecent", minimum(noreg_summary["losses"]))
show_loss!(tv_summary, "TV", minimum(noreg_summary["losses"]))
show_loss!(gr_summary, "GR", minimum(noreg_summary["losses"]))

best_ncc_img, best_nvar_img = get_data(tv_summary)
@vt obj
@vt measured
@vt best_ncc_img
@vt best_nvar_img
@vt res_noreg
