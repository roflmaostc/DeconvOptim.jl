@testset "Testing Main deconvolution code" begin
    Random.seed!(42)

    img = abs.(randn((3, 3)))
    psf = abs.(randn((3, 3)))
    
    res = [4.635723234474804 0.0007095372168799266 5.754390170845983e-14; 0.012691042798718619 1.086315317302081 0.0042021675793670376; 7.328043021151072e-17 0.0023139400413202177 1.7364022862411281] 

    @test all(res .≈ deconvolution(img, psf)[1])
    @test all(res .≈ deconvolution(img, psf, plan_fft=false)[1])
    @test all(res .≈ deconvolution(img, psf, plan_fft=false, iterations=20)[1])

    # testing regularizer
    res2 = [4.188270526990536 5.999388400251461e-10 2.8299849680327642e-8; 1.725273124171714e-7 2.54195544512864 2.0216187854619135e-9; 9.594324085846738e-10 1.2000166997002865e-8 0.7863126081711094] 
    @test all(res2 .≈ deconvolution(img, psf, regularizer=nothing, padding=0.0)[1])
  
    # testing padding
    img = abs.(20 .* randn((3, 3)))
    psf = abs.(randn((3, 3)))
    #= @show deconvolution(img, psf, regularizer=nothing, padding=0.1)[1] =#
    res3 = [2.7358233029549654e-13 91.81808945246226 4.8779432830104286e-17; 2.5136104031366187e-13 89.00977286151101 3.0110340818892347e-13; 3.244894460054766e-12 18.817221108658277 89.77201822876616] 
    @test all(≈(res3, deconvolution(img, psf, regularizer=nothing, padding=0.1)[1], rtol=1e-2))


    # test without mapping
    res4 = [42.27955894415649 52.689377232550505 -24.52278352156622; -39.86494002021505 84.54575838335582 -22.281752192409687; 33.439733220094546 19.407630549139128 59.48869488468371] 
    @test all(res4 .≈ deconvolution(img, psf, regularizer=nothing, padding=0.1, mapping=nothing)[1])


    
    # test broadcasting with image having more dimensions
    img = zeros((3, 3, 2))
    imgc = abs.(randn((3, 3, 1)))
    img[:, :, 1] = imgc
    img[:, :, 2] = imgc
    psf = zeros((3, 3))
    psf[1,1] = 1
   
    res = deconvolution(img, psf, regularizer=GR(num_dims=3, sum_dims=[1,2]))[1]
    @test all(res[:, :, 1] .≈ res[:, :, 2]) 

    img = zeros((3, 3, 2, 1))
    imgc = abs.(randn((3, 3, 1, 1)))
    img[:, :, 1, 1] = imgc
    img[:, :, 2, 1] = imgc
    psf = zeros((3, 3))
    psf[1,1] = 1
   
    res = deconvolution(img, psf, regularizer=GR(num_dims=4, sum_dims=[1,2]))[1]
    @test all(res[:, :, 1, :] .≈ res[:, :, 2, :]) 

end
