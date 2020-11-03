@testset "Forward Models: Convolution" begin

    N = 5
    psf = zeros((N, N))
    psf[1, 1] = 1
    img = randn((N, N))
    
    c(img, psf) = conv_psf(img, psf, [1, 2])
    conv(img, psf) = DeconvOptim.conv_aux(c, img, psf)

    @test conv(img, psf) ≈ img 
    s(img, psf) = sum(conv(img, psf))
    
    @test all(1 .≈ Zygote.gradient(s, img, psf)[1])

end
