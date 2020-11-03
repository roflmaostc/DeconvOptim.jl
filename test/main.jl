@testset "Testing Main deconvolution code" begin
    Random.seed!(42)

    img = randn((3, 3))
    psf = randn((3, 3))
    
    res = [2.2651480070562995 0.37020823409183323 0.60212775719812; 0.002927465797471582 0.012945404132307292 0.1377756221162511; 0.3971985245368018 0.00019987879871227046 0.31339228191542085]
    @test all(res .≈ deconvolution(img, psf)[1])
    @test all(res .≈ deconvolution(img, psf, plan_fft=false)[1])
    @test all(res .≈ deconvolution(img, psf, plan_fft=false, iterations=50)[1])

    res2 = [2.2607906616359643 0.3655684460899124 0.6031252012437225; 0.0026905888577756073 0.013530167239603945 0.13853059609167315; 0.3985361939201093 0.0001444152709048845 0.3084041861830485]
    @test all(res2 .≈ deconvolution(img, psf, regularizer=nothing, padding=0.0)[1])
   
    img = abs.(20 .* randn((3, 3)))
    psf = abs.(randn((3, 3)))
    res3 = [3.244891818080811 17.141098646772267 0.0007561293841754774; 0.034517228270906045 9.034036257026907 0.2063824116147499; 0.0015189996589895396 41.21712794995631 32.96060746230941]
    @show all(res3 .≈ deconvolution(img, psf, regularizer=nothing, padding=0.1)[1])
end
