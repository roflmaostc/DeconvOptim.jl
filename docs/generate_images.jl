using DeconvOptim, TestImages, Images, FFTW, Noise, ImageView
using Plots

function generate_overview()
    
    img = channelview(testimage("resolution_test_512"))
    img ./= maximum(img)
    
    dist = [sqrt((-1 + i - size(img)[1] / 2)^2 + (-1 + j - size(img)[2] / 2)^2)
                for i = 1:size(img)[1],  j = 1:size(img)[2]]
    psf = ifftshift(exp.(-dist .^2 ./ 4.0 .^2))
    psf ./= sum(psf)
    
    img_b = conv_psf(img, psf, [1, 2])
    img_n = poisson(img_b .* 300, 300)
    
   
    reg = DeconvOptim.GR()
    
    @time res, o = deconvolution(img_n, psf, iterations=10,
                                 λ=0.001, lossf=Poisson(), regularizerf=reg)
    @show o
    # simple example of deconvolution
    save("src/assets/img.png", clamp01!(img))
    save("src/assets/img_noisy_index.png", clamp01!(img_n ./ 300))
    save("src/assets/img_rec_index.png", clamp01!(res ./ 300))
    
    
    
    
    img_ft = abs.(fft(img))
    img_ft ./= maximum(img_ft)
    
    img_n_ft = abs.(fft(img_n))
    img_n_ft ./= maximum(img_n_ft)
    
    res_ft = real(fft(res))
    res_ft ./= maximum(res_ft)
    return
end

function norm_fft(x)
    x = fftshift(abs.(fft(x)))
    n = div(size(x)[1], 2)
    x ./= maximum(x)
end



function ideal_freq()
    img = zeros((512, 512))
    for i = 1:1
        img[rand((1:512)), rand((1:512))] = 1
    end
    img = fftshift(img)
    N_phot = 376
    img ./= maximum(img)
    img = convert(Array{Float32}, img .* N_phot)
    dist = [sqrt((-1 + i - size(img)[1] / 2)^2 + (-1 + j - size(img)[2] / 2)^2)
                for i = 1:size(img)[1],  j = 1:size(img)[2]]
    psf = ifftshift(exp.(-dist .^2 ./ 5.0 .^2))
    psf ./= sum(psf)
    psf = convert(Array{Float32}, psf)
    
    #img_b = center_extract(conv_psf(center_set!(copy(z1), img), ifftshift(center_set!(z, fftshift(psf))), [1, 2]), size(img))
    img_b = conv_psf(img, psf, [1, 2])
    img_n = poisson(img_b, N_phot)
    
    reg = DeconvOptim.TV(num_dim=2, sum_dims=[1, 2])
    @time res, o = deconvolution(img_n, psf, iterations=60, λ=0.001f0,
            lossf=Poisson(), regularizerf=reg, padding=0.00, plan_fft=true)
    img_ft = norm_fft(img)[:, 257]
    img_n_ft = norm_fft(img_n)[:, 257]
    res_ft = norm_fft(res)[:, 257]
    freq = fftshift(fftfreq(512, 1))
    mtf = norm_fft(psf)[:, 257]
    plot(freq, mtf, xlabel="Frequency in 1/pixel", ylabel = "Normalized intensity in Frequency space", label="OTF", dpi=300)
    plot!(freq, img_ft,label="Image with Constant Frequency content")
    plot!(freq, img_n_ft, label="Blurry image with noise", linestyle = :dot)
    plot!(freq, res_ft, label="Deconvolved image")
    
    savefig("src/assets/ideal_frequencies.png")
end

generate_overview()
ideal_freq()
