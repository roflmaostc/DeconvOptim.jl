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
    res3 = [3.700346192004195 16.163004402781457 0.0005317269571170434; 0.3852386808335988 14.378882057906575 0.04691706650167405; 1.5059695704739982 16.94303953714345 22.72731111751148] 
    @test all(≈(res3, deconvolution(img, psf, regularizer=nothing, padding=0.1)[1], rtol=1e-2))


    # test without mapping
    res4 = [-13.085096066984729 34.39174935297922 -11.353417020874208; -13.01079706662783 43.76596781851713 -7.565144283495296; 16.740081837985805 -7.488374542587605 37.666978022259336] 
    #= @show deconvolution(img, psf, regularizer=nothing, padding=0.1, mapping=nothing)[1] =# 
    @test all(≈(res4, deconvolution(img, psf, regularizer=nothing, padding=0.1, mapping=nothing)[1], rtol=0.1))

    # test OptimPackNextGen optimizers  (which is currently not officially released yet)
    res5 = [0.49633      16.4936   0.00862045; 0.0139499    15.8587   2.31716; 0.000227487   9.263   19.5289]
    @test all(≈(res5, deconvolution(img, psf, opt=vmlmb!, opt_options=(mem=20, lower=0, lnsrch=OptimPackNextGen.LineSearches.MoreThuenteLineSearch()),
    regularizer=nothing, padding=0.1, mapping=nothing, opt_package=Opt_OptimPackNextGen)[1], rtol=0.1))
    
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
