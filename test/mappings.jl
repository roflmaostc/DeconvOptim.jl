

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

    function f(x)
        @test all(.≈(Zygote.gradient(x -> sum(p(x)), x)[1], (p(x .+ 1e-8) .- p(x))./1e-8, rtol=1e-4))
        @test Zygote.gradient(x -> sum(p(x)), x)[1] ≈ DeconvOptim.f_pw_pos_grad(x)
    end

    f([1.1, 12312.2, -10.123, 22.2, -123.23, 0])
end

@testset "Abs_positive" begin
    p, p_inv = Abs_positive()

    x = abs.(randn((10, 10)))
    x2 = 100 .*randn((10, 10))

    @test x ≈ p(p_inv(x))
    @test x ≈ p_inv(p(x))

    @test all(p(x2) .>= 0)
end
