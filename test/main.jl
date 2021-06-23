@testset "Testing Main deconvolution code" begin
    Random.seed!(42)

    img = [0.5560268761463861 0.29948409035891055 0.46860588216767457; 0.444383357109696 1.7778610980573246 0.15614346264074028; 0.027155338009193845 1.14490153172882 2.641991008076796]
    psf = [1.0033099014594844 0.5181487878771377 0.8862052960481365; 1.0823812056084292 1.4913791170403063 0.6845647041648603; 0.18702790710363 0.3675627461748204 1.590579974922555]

    
    res = [2.4393835034275493 0.013696697097634842 0.0002833052222499294; 0.07541628019133978 1.0066536888249171 0.02222160874466724; 0.0004945773667781262 0.008547708184955495 3.717245734531717] 

    #@show deconvolution(img, psf, λ=0.01)[1]
    @test all(≈(res, deconvolution(img, psf, λ=0.01)[1], rtol=0.1))
    @test all(≈(res, deconvolution(img, psf, λ=0.01)[1], rtol=0.1))
    @test all(≈(res, deconvolution(img, psf, λ=0.01, iterations=20)[1], rtol=0.1))

    # testing regularizer
    res2 = [4.188270526990536 5.999388400251461e-10 2.8299849680327642e-8; 1.725273124171714e-7 2.54195544512864 2.0216187854619135e-9; 9.594324085846738e-10 1.2000166997002865e-8 0.7863126081711094] 
    @test all(≈(res2, deconvolution(img, psf, regularizer=nothing, padding=0.0)[1], rtol=0.1))
  
    # testing padding
    img = [8.21306764808666 10.041589152470781 86.74936458947307; 17.126996611046078 4.324960146596254 11.39657297820361; 21.019754207225656 14.128485444028698 28.441178191470662]

    psf = [0.37240087577993225 0.562668812321259 0.6810849274435286; 0.36901028455183293 0.10686911035365092 1.3391251213773154; 0.007612980079313577 0.5694584949295476 0.23828371819888622]

    #@show deconvolution(img, psf, regularizer=nothing, padding=0.1)[1]
    res3 = [1.0406577489364471e-7 91.81363444807677 2.272608189988653e-7; 1.2703220368343009e-7 89.01393312854161 5.487620214472806e-8; 3.5819990308679853e-9 18.814647865228448 89.76902076477796] 
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
    img_b = conv(img, psf);
    img_n = poisson(img_b, 300);

    reg = GR()
    res = richardson_lucy_iterative(img_n, psf, regularizer=reg, iterations=200);
    res2, o = deconvolution(img_n, psf, regularizer=reg, iterations=30);
    @test ≈(res .+ 1, res2 .+ 1, rtol=0.003)
    
    reg = TV()
    res = richardson_lucy_iterative(img_n, psf, iterations=500, λ=0.005, regularizer=reg);
    res2, o = deconvolution(img_n, psf, iterations=35, regularizer=reg, λ=0.005);
    @test ≈(res2 .+1, res .+ 1, rtol=0.02)

    reg = nothing 
    res = richardson_lucy_iterative(img_n, psf, regularizer=reg, iterations=400);
    res2, o = deconvolution(img_n, psf, regularizer=reg, iterations=40);
    @test ≈(res2 .+1, res .+ 1, rtol=0.02)


end
