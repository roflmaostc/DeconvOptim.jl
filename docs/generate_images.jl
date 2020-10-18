using DeconvOptim, TestImages, Images, FFTW, Noise

img = channelview(testimage("resolution_test_512"))

dist = [sqrt((-1 + i - size(img)[1] / 2)^2 + (-1 + j - size(img)[2] / 2)^2)
            for i = 1:size(img)[1],  j = 1:size(img)[2]]
psf = ifftshift(exp.(-dist .^2 ./ 4.0 .^2))
psf ./= sum(psf)

img_b = conv_psf(img, psf, [1, 2])
img_n = poisson(img_b, 300)

img_n ./= maximum(img_n)

reg = DeconvOptim.GR(λ=0.001)
@time res, o = deconvolution(img_n, psf, iterations=10,
        lossf=Poisson(), regularizerf=reg)


save("docs/src/assets/img_noisy_index.png", clamp01!(img_n))
save("docs/src/assets/img_rec_index.png", clamp01!(res[:, :, 1]))
