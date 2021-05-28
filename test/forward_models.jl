@testset "Forward Models: Convolution" begin

    N = 5
    psf = zeros((N, N))
    psf[1, 1] = 1
    img = randn((N, N))
    
    c(img, psf) = conv(img, psf, [1, 2])
    conv_temp(img, psf) = DeconvOptim.conv_aux(c, img, psf)

    @test conv_temp(img, psf) ≈ img 
    s(img, psf) = sum(conv_temp(img, psf))
    
    @test all(1 .≈ Zygote.gradient(s, img, psf)[1])

end
