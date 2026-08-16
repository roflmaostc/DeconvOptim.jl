
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

@testset "HS sum_dims" begin
    x3 = abs.(randn(Float64, (6, 6, 6)))
    x4 = abs.(randn(Float64, (6, 6, 6, 6)))

    # sum_dims=(1,2) on a 3D array == sum over the per-slice 2D HS
    hs12 = HS(sum_dims=(1, 2))
    hs12d2 = HS()
    @test hs12(x3) ≈ sum(k -> hs12d2(view(x3, :, :, k)), axes(x3, 3))

    # p=1 vs the default HS (p=1) must agree
    @test HS(p=1)(x3) ≈ sum(k -> hs12d2(view(x3, :, :, k)), axes(x3, 3))

    # sum_dims=(2,3) == per-slice HS over the slices holding dims 2 and 3
    hs23 = HS(sum_dims=(2, 3))
    @test hs23(x3) ≈ sum(i -> hs12d2(view(x3, i, :, :)), axes(x3, 1))

    # sum_dims=(1,3) == per-slice HS over the slices holding dims 1 and 3
    hs13 = HS(sum_dims=(1, 3))
    @test hs13(x3) ≈ sum(j -> hs12d2(view(x3, :, j, :)), axes(x3, 2))

    # general p path on a 3D array with sum_dims
    hsp2 = HS(sum_dims=(1, 2), p=2)
    @test hsp2(x3) ≈ sum(k -> DeconvOptim.HSp(view(x3, :, :, k), p=2), axes(x3, 3))

    # weights positional: H11' = w1^2*H11, H22' = w2^2*H22, H12' = w1*w2*H12
    hsw = HS(sum_dims=(1, 2), weights=[2.0, 3.0])
    rw = 0.0
    for k in axes(x3, 3), j in 1:4, i in 1:4
        h11 = x3[i+2, j, k] - 2*x3[i+1, j, k] + x3[i, j, k]
        h22 = x3[i, j+2, k] - 2*x3[i, j+1, k] + x3[i, j, k]
        rw += abs(1f-8 + 4*h11 + 9*h22)
    end
    @test hsw(x3) ≈ rw

    # general p with weights on a 3D array
    hsp2w = HS(sum_dims=(1, 2), p=2, weights=[2.0, 3.0])
    rp2w = 0.0
    for k in axes(x3, 3), j in 1:4, i in 1:4
        h11 = x3[i+2, j, k] - 2*x3[i+1, j, k] + x3[i, j, k]
        h22 = x3[i, j+2, k] - 2*x3[i, j+1, k] + x3[i, j, k]
        h12 = x3[i+1, j+1, k] - x3[i+1, j, k] - x3[i, j+1, k] + x3[i, j, k]
        a = 4*h11; d = 9*h22; b = 6*h12
        r = sqrt(1f-8 + (a-d)^2 + 4*b^2)
        lam1 = 0.5*(a+d+r); lam2 = 0.5*(a+d-r)
        rp2w += abs(1f-8 + lam1^2 + lam2^2)^0.5
    end
    @test hsp2w(x3) ≈ rp2w

    # sum_dims on a 4D array == per-slice 2D HS over the (dim1, dim2) slices
    hs12_4 = HS(sum_dims=(1, 2))
    @test hs12_4(x4) ≈ sum(hs12d2(view(x4, :, :, k, l)) for k in axes(x4, 3), l in axes(x4, 4))

    # sum_dims=(1,3) on a 4D array == per-slice 2D HS over the (dim1, dim3) slices
    hs13_4 = HS(sum_dims=(1, 3))
    ref13_4 = sum(hs12d2(view(x4, :, j, :, l)) for j in axes(x4, 2), l in axes(x4, 4))
    @test hs13_4(x4) ≈ ref13_4

    # exactly two distinct sum_dims must be given, else throw
    @test_throws ArgumentError HS(sum_dims=(1, 2, 3))(x3)
    @test_throws ArgumentError HS(sum_dims=(1, 1))(x3)
    @test_throws ArgumentError HS(sum_dims=(1,))(x3)
    @test_throws ArgumentError HS(sum_dims=(1, 4))(x3)
end
