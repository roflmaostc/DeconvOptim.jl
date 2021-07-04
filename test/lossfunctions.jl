@testset "Poisson loss" begin
    N = 6
    img = abs.(randn((N, N)))

    @test 0.22741127776021886 ≈ poisson_aux([1.,2.], [3.,4.]) 

    @test all(0 .≈ Zygote.gradient(poisson_aux, img, img)[1])
    
    @test all( -1 .≈ Zygote.gradient(poisson_aux, [1.], [2.])[1])
end


@testset "Gaussian Loss" begin
    N = 6
    img = abs.(randn((N, N)))

    gauss = Gauss()
    @test 0 ≈ gauss(img, img)

    @test 0 ≈ gauss_aux(img, img)
    
    @test all(0 .≈ Zygote.gradient(gauss_aux, img, img)[1])

end


@testset "Scaled Gauss" begin
    N = 6
    img = abs.(randn((N, N)))
    
    scaled_gauss = ScaledGauss(0)

    @test 3.4094379124341003 ≈ scaled_gauss([5.], [2.])
    @test 3.4094379124341003 ≈ scaled_gauss_aux([5.], [2.], read_var=0)

    @test all( -0.75 .≈ Zygote.gradient((a, b) -> scaled_gauss_aux(a, b, read_var=1), [1.], [2.])[1])
end

@testset "Anscombe Loss" begin
    N = 6
    img = abs.(randn((N, N)))

    anscombe = Anscombe(1.0)
    @test 0 ≈ anscombe(img, img)

    @test 0 ≈ anscombe_aux(img, img,b=1)

    @test all(0 .≈ Zygote.gradient(im1 -> anscombe_aux(im1,img, b=1), img)[1])

end
