@testset "Testing Main deconvolution code" begin
    Random.seed!(42)

    img = randn((3, 3))
    psf = randn((3, 3))
    
    res = [2.2651480070562995 0.37020823409183323 0.60212775719812; 0.002927465797471582 0.012945404132307292 0.1377756221162511; 0.3971985245368018 0.00019987879871227046 0.31339228191542085]
    @test all(res .≈ deconvolution(img, psf)[1])
    @test all(res .≈ deconvolution(img, psf, plan_fft=false)[1])
    @test all(res .≈ deconvolution(img, psf, plan_fft=false, iterations=50)[1])

    # testing regularizer
    res2 = [2.2607906616359643 0.3655684460899124 0.6031252012437225; 0.0026905888577756073 0.013530167239603945 0.13853059609167315; 0.3985361939201093 0.0001444152709048845 0.3084041861830485]
    @test all(res2 .≈ deconvolution(img, psf, regularizer=nothing, padding=0.0)[1])
   
    # testing padding
    img = abs.(20 .* randn((3, 3)))
    psf = abs.(randn((3, 3)))
    res3 = [3.244891818080811 17.141098646772267 0.0007561293841754774; 0.034517228270906045 9.034036257026907 0.2063824116147499; 0.0015189996589895396 41.21712794995631 32.96060746230941]
    @test all(res3 .≈ deconvolution(img, psf, regularizer=nothing, padding=0.1)[1])


    # test without mapping
    res4 = [-19.24648922082771 39.61274950533089 -15.220294412294914; -18.611821434506325 59.7605384098004 1.8270388012669567; 9.25158195360382 -4.14983150380469 33.25226235037082]
    @test all(res4 .≈ deconvolution(img, psf, regularizer=nothing, padding=0.1, mapping=nothing)[1])


    
    # test broadcasting with image having more dimensions
    img = zeros((3, 3, 2))
    imgc = randn((3, 3, 1))
    img[:, :, 1] = imgc
    img[:, :, 2] = imgc
    psf = zeros((3, 3))
    psf[1,1] = 1
   
    res = deconvolution(img, psf, regularizer=GR(num_dims=3, sum_dims=[1,2]))[1]
    @test all(res[:, :, 1] .≈ res[:, :, 2]) 

    img = zeros((3, 3, 2, 1))
    imgc = randn((3, 3, 1, 1))
    img[:, :, 1, 1] = imgc
    img[:, :, 2, 1] = imgc
    psf = zeros((3, 3))
    psf[1,1] = 1
   
    res = deconvolution(img, psf, regularizer=GR(num_dims=4, sum_dims=[1,2]))[1]
    @test all(res[:, :, 1, :] .≈ res[:, :, 2, :]) 

end
