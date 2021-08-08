using DeconvOptim, TestImages, Images, FFTW, Noise, ImageView
using Plots

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
    
    #img_b = center_extract(conv(center_set!(copy(z1), img), ifftshift(center_set!(z, fftshift(psf))), [1, 2]), size(img))
    img_b = conv(img, psf, [1, 2])
    img_n = poisson(img_b, N_phot)
    
    reg = DeconvOptim.TV(num_dims=2, sum_dims=[1, 2])
    @time res, o = deconvolution(img_n, psf, iterations=10, λ=0.001f0,
            loss=Poisson(), regularizer=reg, padding=0.00, plan_fft=true)
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

ideal_freq()
