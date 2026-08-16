# here we compare various deconvolution options in terms of image quality
# add SyntheticObjects, NDTools, CUDA
using DeconvOptim, SyntheticObjects, Random, Noise, NDTools
using CUDA
using View5D

function main()
        sz = (128, 128, 128)
        obj = 100f0 .* filaments3D(sz);

        # simulate a simple PSF
        sz = size(obj); 
        R_max = sz[1] ./ 12.0;
        zz = reorient(-sz[3]÷2:sz[3]÷2-1, Val(3))
        psf = DeconvOptim.ifftshift(DeconvOptim.fftshift(generate_psf(sz[1:2], R_max), (1,2)) .* exp.(-abs2.(zz./15f0)));

        # simulate a perfect image
        conv_img = DeconvOptim.conv(obj, psf);

        # set a fixed point for the measured data quality
        max_photons = 1000

        Random.seed!(42)
        measured = poisson(conv_img, max_photons);

        # opt_options = nothing
        # opt_options, noreg_summary = DeconvOptim.options_trace_deconv(obj, iterations, Non_negative());

        use_cuda = true
        # use_cuda = false
        psf = (use_cuda) ? cu(psf) : Array(psf);
        measured = (use_cuda) ? cu(measured) : Array(measured);

        iterations =  100
        CUDA.reclaim()
        R = nothing
        CUDA.@time @CUDA.sync res_noreg = deconvolution(measured, psf; mapping=Non_negative(), regularizer=R, iterations=iterations);
        @time res_noreg = deconvolution(measured, psf; mapping=Non_negative(), regularizer=R, iterations=iterations);
        # NoReg: CUDA: 2.16 sec, CPU: 22.40 sec
        # OldV0.7.4, NoReg: CUDA: 2.28 sec, CPU: 26.21 sec
        
        CUDA.reclaim()
        # R = GR(num_dims=3)
        R = GR()
        CUDA.@time @CUDA.sync res_gr = deconvolution(measured, psf; mapping=Non_negative(), regularizer = R, iterations=iterations);
        @time res_gr = deconvolution(measured, psf; mapping=Non_negative(), regularizer = R, iterations=iterations);
        # GR(): CUDA: 1.62 sec, CPU: 13.96 sec, CPU view version: 24.35 sec
        # OldV0.7.4, GR(): CUDA: (error) sec, CPU: 35.87 sec, CPU view version: (not existing) sec  (only with num_dims=3)

        CUDA.reclaim()
        R = TV() # num_dims=3
        # R = TV(num_dims=3) # num_dims=3
        CUDA.@time @CUDA.sync res_tv = deconvolution(measured, psf; mapping=Non_negative(), regularizer = R, iterations=iterations);
        @time res_tv = deconvolution(measured, psf; mapping=Non_negative(), regularizer = R, iterations=iterations);
        # TV(): CUDA: 2.9 sec, CPU: 21.91 sec, CPU view version: 48.04 sec 
        # OldV0.7.4, TV(): CUDA: 3.21 sec, CPU: 32.43 sec, CPU view version: 82.82 sec  (only with num_dims=3)

        CUDA.reclaim()
        # R = TH(num_dims=3)
        R = TH()
        CUDA.@time @CUDA.sync res_th = deconvolution(measured, psf; mapping=Non_negative(), regularizer = R, iterations=iterations);
        @time res_th = deconvolution(measured, psf; mapping=Non_negative(), regularizer = R, iterations=iterations);
        # TH(): CUDA: 4.9 sec, CPU: 23.78 sec, CPU view version: 88 sec
        # OldV0.7.4, TH(): CUDA: (error) sec, CPU: 31.11 sec, CPU view version: (not existing) sec  (only with num_dims=3)

        CUDA.reclaim()
        # R = TH(num_dims=3)
        R = HS() # p=1
        # R = DeconvOptim.HS_cuda()
        CUDA.@time @CUDA.sync res_hs = deconvolution(measured, psf; mapping=Non_negative(), regularizer = R, iterations=iterations);
        @time res_hs = deconvolution(measured, psf; mapping=Non_negative(), regularizer = R, iterations=iterations);
        # HS(): CUDA: 4.23 sec, CPU: 68 sec, CPU view version: (not existing)
        # OldV0.7.4, TH(): CUDA: (error) sec, CPU: 31.11 sec, CPU view version: (not existing) (only with num_dims=3)

        @vt obj
        @vt measured
        @vt res_noreg
        @vt res_gr
        @vt res_hs
        @vt res_th

end
