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
    @test 0 ≈ gauss_aux(img, img)
    
    @test all(0 .≈ Zygote.gradient(gauss_aux, img, img)[1])

end


@testset "Scaled Gauss" begin
    N = 6
    img = abs.(randn((N, N)))
    
    @test 1.7047189562170502 ≈ scaled_gauss_aux([5.], [2.]) 
    
    @test all( -1 .≈ Zygote.gradient(scaled_gauss_aux, [1.], [2.])[1])


end
