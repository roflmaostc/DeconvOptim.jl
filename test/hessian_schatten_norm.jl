
@testset "Test eigvals" begin
    
    function f(a1::T,b1,c1) where T
        a = Array{T, 2}(undef, 1, 1)
        a[1,1] = a1
        b = Array{T, 2}(undef, 1, 1)
        b[1,1] = b1
        c = Array{T, 2}(undef, 1, 1)
        c[1,1] = c1
        @test all(.≈(DeconvOptim.eigvals_symmetric_tullio(a,b,c), DeconvOptim.eigvals_symmetric(a,b,c)))
    end

    f(10.0, 20.0, -10.0)
    f(0f0, -12f0, 13f0)
end


@testset "Schatten norm consistent" begin

    x = [1 2 3; 1 1 1; 0 0 -1f0]
    @test DeconvOptim.HSp(x, p = 1) ≈ 0.9999999900000001
    @test DeconvOptim.HSp(x, p = 2) ≈ 1.732050831647567
    @test abs.(DeconvOptim.HSp(x, p = 1)) ≈ DeconvOptim.HS1(x) 
end
