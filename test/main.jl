@testset "Testing Main deconvolution code" begin
    Random.seed!(42)

    img = abs.(randn((3, 3)))
    psf = abs.(randn((3, 3)))
    
    res = [4.933040319558362 1.640892951656587e-9 1.941347101658743e-8; 1.4783685597711925e-8 4.723499796776938e-7 3.668727889064141e-9; 2.765081805017183e-9 3.280055085010056e-9 2.583774989955379] 

    #= @show deconvolution(img, psf, λ=0.01)[1] =#
    @test all(≈(res, deconvolution(img, psf, λ=0.01)[1], rtol=0.1))
    @test all(≈(res, deconvolution(img, psf, plan_fft=false, λ=0.01)[1], rtol=0.1))
    @test all(≈(res, deconvolution(img, psf, plan_fft=false, λ=0.01, iterations=20)[1], rtol=0.1))

    # testing regularizer
    res2 = [4.188270526990536 5.999388400251461e-10 2.8299849680327642e-8; 1.725273124171714e-7 2.54195544512864 2.0216187854619135e-9; 9.594324085846738e-10 1.2000166997002865e-8 0.7863126081711094] 
    @test all(≈(res2, deconvolution(img, psf, regularizer=nothing, padding=0.0)[1], rtol=0.1))
  
    # testing padding
    img = abs.(20 .* randn((3, 3)))
    psf = abs.(randn((3, 3)))
    #= @show deconvolution(img, psf, regularizer=nothing, padding=0.1)[1] =#
    res3 = [2.7358233029549654e-13 91.81808945246226 4.8779432830104286e-17; 2.5136104031366187e-13 89.00977286151101 3.0110340818892347e-13; 3.244894460054766e-12 18.817221108658277 89.77201822876616] 
    @test all(≈(res3, deconvolution(img, psf, regularizer=nothing, padding=0.1)[1], rtol=1e-2))


    # test without mapping
    res4 = [-5.332320754919289 40.63633433786456 -18.173096688407522; -15.597390722842547 59.01033285248438 -4.122592175577954; 13.402949467681973 6.4622595721116465 42.82143693411391] 
    #= @show deconvolution(img, psf, regularizer=nothing, padding=0.1, mapping=nothing)[1] =# 
    @test all(≈(res4, deconvolution(img, psf, regularizer=nothing, padding=0.1, mapping=nothing)[1], rtol=0.1))


    
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
    @test all(≈(res[:, :, 1, :], res[:, :, 2, :], rtol=0.1))

end




@testset "Compare optimization with iterative lucy richardson scheme" begin

    img = Float32.(testimage("resolution_test_512"));
    psf = Float32.(generate_psf(size(img), 30));
    img_b = conv_psf(img, psf);
    img_n = poisson(img_b, 300);

    reg = GR()
    res = richardson_lucy_iterative(img_n, psf, regularizer=reg, iterations=200);
    res2, o = deconvolution(img_n, psf, regularizer=reg, iterations=20);
    @test res .+ 1 ≈ res2 .+ 1
    
    reg = TV()
    res = richardson_lucy_iterative(img_n, psf, iterations=500, λ=0.005, regularizer=reg);
    res2, o = deconvolution(img_n, psf, iterations=35, regularizer=reg, λ=0.005);
    @test ≈(res2 .+1, res .+ 1, rtol=0.02)

    reg = nothing 
    res = richardson_lucy_iterative(img_n, psf, regularizer=reg, iterations=400);
    res2, o = deconvolution(img_n, psf, regularizer=reg, iterations=40);
    @test ≈(res2 .+1, res .+ 1, rtol=0.02)


end
