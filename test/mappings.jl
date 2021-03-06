

@testset "Non_negative" begin

    p, p_inv = Non_negative()

    x = abs.(randn((10, 10)))
    x2 = 100 .*randn((10, 10))

    @test x ≈ p(p_inv(x))
    @test x ≈ p_inv(p(x))

    @test all(p(x2) .>= 0)
end

@testset "Map_0_1" begin

    p, p_inv = Map_0_1()

    x = abs.(randn((10, 10)))
    x2 = 100 .*randn((10, 10))

    @test x ≈ p_inv(p(x))

    @test all(p(x2) .>= 0)
    @test all(p(x2) .<= 1)
end

@testset "Pow4_positive" begin
    p, p_inv = Pow4_positive()

    x = abs.(randn((10, 10)))
    x2 = 100 .*randn((10, 10))

    @test x ≈ p(p_inv(x))
    @test x ≈ p_inv(p(x))

    @test all(p(x2) .>= 0)
end

@testset "Piecewise_positive" begin
    p, p_inv = Piecewise_positive()

    x = abs.(randn((10, 10)))
    x2 = 100 .*randn((10, 10))

    @test x ≈ p(p_inv(x))
    @test x ≈ p_inv(p(x))

    @test all(p(x2) .>= 0)
end

@testset "Abs_positive" begin
    p, p_inv = Abs_positive()

    x = abs.(randn((10, 10)))
    x2 = 100 .*randn((10, 10))

    @test x ≈ p(p_inv(x))
    @test x ≈ p_inv(p(x))

    @test all(p(x2) .>= 0)
end
