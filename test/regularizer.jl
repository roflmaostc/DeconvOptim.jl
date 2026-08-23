@testset "generate indices" begin
    @test DeconvOptim.generate_indices(5, 2, 1, 5) == (Any[:i1, :(i2 + 1), :i3, :i4, :i5], Any[:i1, :(i2 + 5), :i3, :i4, :i5])
end

@testset "generate_laplace" begin
    x = DeconvOptim.generate_laplace(2, [1, 2], [4 , 5], debug=true)
    @test x==Any[:(res = abs2((4 * arr[i1 + 1, i2] + 4 * arr[i1 + -1, i2]) + (5 * arr[i1, i2 + 1] + 5 * arr[i1, i2 + -1]) + -(18* arr[i1, i2])))]
    x = DeconvOptim.generate_laplace(2, [1, 2], [1 , 1], debug=true)
    @test x==Any[:(res = abs2((1 * arr[i1 + 1, i2] + 1 * arr[i1 + -1, i2]) + (1 * arr[i1, i2 + 1] + 1 * arr[i1, i2 + -1]) + -(4 * arr[i1, i2])))]
    
end

@testset "Tikhonov" begin
    x = [1,2,3,1,3,1,12.0,2,2,3,2.0]
    reg = Tikhonov(num_dims=1, sum_dims=[1], weights=[1])
    @test 756 ≈ reg(x)

    reg = Tikhonov(num_dims=1, mode="spatial_grad_square")
    @test 188 ≈ reg(x)
    
    reg = Tikhonov(num_dims=1, mode="identity")
    @test 190 ≈ reg(x)

    arr = abs.(randn(Float64, (7, 6, 5)))

    # Tikhonov_cuda (view-based) matches the CPU @tullio kernel for all modes
    # (build into a variable first, to avoid the @eval world-age flake)
    tk1 = Tikhonov(num_dims=1, sum_dims=[1], weights=[1])
    tk1c = DeconvOptim.Tikhonov_cuda(num_dims=1, sum_dims=[1], weights=[1])
    @test tk1c(x) ≈ tk1(x)

    arr2d = abs.(randn(Float64, (7, 6)))

    tk2 = Tikhonov(num_dims=2, sum_dims=[1, 2], weights=[1, 2], mode="spatial_grad_square", step=2)
    tk2c = DeconvOptim.Tikhonov_cuda(num_dims=2, sum_dims=[1, 2], weights=[1, 2], mode="spatial_grad_square", step=2)
    @test tk2c(arr2d) ≈ tk2(arr2d)

    tk3 = Tikhonov(num_dims=3, weights=[1.0, 2.0, 3.0], step=2)
    tk3c = DeconvOptim.Tikhonov_cuda(num_dims=3, weights=[1.0, 2.0, 3.0], step=2)
    @test tk3c(arr) ≈ tk3(arr)

    tk4 = Tikhonov(num_dims=3, sum_dims=[1, 3], weights=[1.0, 3.0], mode="spatial_grad_square")
    tk4c = DeconvOptim.Tikhonov_cuda(num_dims=3, sum_dims=[1, 3], weights=[1.0, 3.0], mode="spatial_grad_square")
    @test tk4c(arr) ≈ tk4(arr)

    # identity mode is the same on CPU and CUDA
    tk5 = Tikhonov(num_dims=3, mode="identity")
    tk5c = DeconvOptim.Tikhonov_cuda(num_dims=3, mode="identity")
    @test tk5(arr) ≈ tk5c(arr)
end

@testset "Good's roughness" begin

    x = generate_GR(5, [1,2], [4, 5], 1, -1, debug=true)

    @test x == Any[:(res = -2.0 * arr[i1, i2, i3, i4, i5] * (4 * (arr[i1 + 1, i2, i3, i4, i5] + arr[i1 + -1, i2, i3, i4, i5]) + 5 * (arr[i1, i2 + 1, i3, i4, i5] + arr[i1, i2 + -1, i3, i4, i5]) + -18 * arr[i1, i2, i3, i4, i5]))]

    x = [1,2,3,1,3,1,12.0,2,2,3,2.0]
    reg = GR(num_dims=1, sum_dims=[1], weights=[1])
    @test 22.71233466779126 ≈ reg(x)


end


@testset "GR_cuda" begin
    x = [1,2,3,1,3,1,12.0,2,2,3,2.0]
    reg_cpu = GR(num_dims=1, sum_dims=[1], weights=[1])
    reg_gpu = DeconvOptim.GR_cuda(num_dims=1, sum_dims=[1], weights=[1])
    @test reg_gpu(x) ≈ reg_cpu(x)

    arr = abs.(randn(Float64, (6, 5, 7)))

    reg_cpu = GR(num_dims=3, sum_dims=[1, 2, 3], weights=[1, 1, 2], mode="forward", step=1)
    reg_gpu = DeconvOptim.GR_cuda(num_dims=3, sum_dims=[1, 2, 3], weights=[1, 1, 2], mode="forward", step=1)
    @test reg_gpu(arr) ≈ reg_cpu(arr)

    reg_cpu = GR(num_dims=3, sum_dims=[1, 2, 3], weights=[1, 1, 2], mode="central", step=1)
    reg_gpu = DeconvOptim.GR_cuda(num_dims=3, sum_dims=[1, 2, 3], weights=[1, 1, 2], mode="central", step=1)
    @test reg_gpu(arr) ≈ reg_cpu(arr)

    reg_cpu = GR(num_dims=3, sum_dims=[1, 2, 3], weights=[1, 1, 2], mode="forward", step=2)
    reg_gpu = DeconvOptim.GR_cuda(num_dims=3, sum_dims=[1, 2, 3], weights=[1, 1, 2], mode="forward", step=2)
    @test reg_gpu(arr) ≈ reg_cpu(arr)

    # partial sum dimensions
    arr4 = abs.(randn(Float64, (3, 5, 4, 3)))
    reg_cpu = GR(num_dims=4, sum_dims=[1, 2], weights=[1, 1], mode="forward")
    reg_gpu = DeconvOptim.GR_cuda(num_dims=4, sum_dims=[1, 2], weights=[1, 1], mode="forward")
    @test reg_gpu(arr4) ≈ reg_cpu(arr4)

    # central mode and step>1 with non-uniform weights (view kernel == @tullio)
    reg_cpu = GR(num_dims=3, sum_dims=[1, 2, 3], weights=[1.0, 2.0, 3.0], mode="central", step=2)
    reg_gpu = DeconvOptim.GR_cuda(num_dims=3, sum_dims=[1, 2, 3], weights=[1.0, 2.0, 3.0], mode="central", step=2)
    @test reg_gpu(arr) ≈ reg_cpu(arr)

    # explicit num_dims both forward and central match the view kernels
    reg_cpu = GR(num_dims=3, weights=[1.0, 2.0, 3.0], mode="central")
    reg_gpu = DeconvOptim.GR_cuda(num_dims=3, weights=[1.0, 2.0, 3.0], mode="central")
    @test reg_gpu(arr) ≈ reg_cpu(arr)
end


@testset "TV" begin
    x = [1,2,3,1,3,1,12.0,2,2,3,2.0]
    reg = TV(num_dims=1, sum_dims=[1], weights=[1])
    @test 31.00010002845424 ≈ reg(x)

    # tests would fail:
    # @test DeconvOptim.TV_cuda(num_dims=2)(x) ≈ reg(x)
    # @test DeconvOptim.TV_cuda(num_dims=3)(x) ≈ reg(x)

    x = generate_TV(4, [1,2], [5, 7], 1, -1, debug=true)
    @test x == Any[:(res = sqrt(5 * abs2(arr[i1 + 1, i2, i3, i4] - arr[i1 + -1, i2, i3, i4]) + 7 * abs2(arr[i1, i2 + 1, i3, i4] - arr[i1, i2 + -1, i3, i4]) + 1.0f-8))]

    arr = abs.(randn(Float64, (8, 9, 10)))

    # TV view-based kernel (TV_cuda) matches the CPU @tullio kernel for
    # central mode, custom step and subset sum_dims
    # (build into a variable first, to avoid the @eval world-age flake)
    tv1 = TV(num_dims=3, mode="forward", step=2, weights=[1.0, 2.0, 3.0])
    tv1c = DeconvOptim.TV_cuda(num_dims=3, mode="forward", step=2, weights=[1.0, 2.0, 3.0])
    @test tv1(arr) ≈ tv1c(arr)

    tv2 = TV(num_dims=3, mode="central", weights=[1.0, 2.0, 3.0])
    tv2c = DeconvOptim.TV_cuda(num_dims=3, mode="central", weights=[1.0, 2.0, 3.0])
    @test tv2(arr) ≈ tv2c(arr)

    tv3 = TV(num_dims=3, sum_dims=[1, 3], weights=[1.0, 3.0])
    tv3c = DeconvOptim.TV_cuda(num_dims=3, sum_dims=[1, 3], weights=[1.0, 3.0])
    @test tv3(arr) ≈ tv3c(arr)

    tv4 = TV(num_dims=3, mode="central", step=2)
    tv4c = DeconvOptim.TV_cuda(num_dims=3, mode="central", step=2)
    @test tv4(arr) ≈ tv4c(arr)

end


@testset "TH" begin
    a1 = abs.(randn(Float64, 7))
    th_cpu1 = TH(num_dims=1)
    th_gpu1 = DeconvOptim.TH_cuda(num_dims=1)
    @test th_gpu1(a1) ≈ th_cpu1(a1)

    a2 = abs.(randn(Float64, (7, 7)))
    th_cpu2 = TH(num_dims=2)
    th_gpu2 = DeconvOptim.TH_cuda(num_dims=2)
    @test th_gpu2(a2) ≈ th_cpu2(a2)

    a3 = abs.(randn(Float64, (7, 7, 7)))
    th_cpu3 = TH(num_dims=3)
    th_gpu3 = DeconvOptim.TH_cuda(num_dims=3)
    @test th_gpu3(a3) ≈ th_cpu3(a3)
end

@testset "auto num_dims" begin
    x1 = abs.(randn(Float64, 9))
    x2 = abs.(randn(Float64, (6, 7)))
    x3 = abs.(randn(Float64, (5, 6, 7)))
    x4 = abs.(randn(Float64, (4, 5, 6, 7)))

    # GR: num_dims=nothing selects the same explicit config on CPU and view-based GPU path
    # NOTE: the auto regularizers must be built into a variable before use, since
    # the CPU path uses @eval-generated @tullio closures (world-age flake otherwise)
    gr_auto = GR()
    gr_auto_c = GR(mode="central")
    gr_cpu1 = GR(num_dims=1, weights=[1])
    gr_cpu2 = GR(num_dims=2, weights=[1, 1])
    gr_cpu3 = GR(num_dims=3, weights=[1, 1, 1])
    gr_cpu4 = GR(num_dims=4, weights=[1, 1, 1, 1])
    gr_cpu3c = GR(num_dims=3, weights=[1, 1, 1], mode="central")
    @test gr_auto(x1) ≈ gr_cpu1(x1)
    @test gr_auto(x2) ≈ gr_cpu2(x2)
    @test gr_auto(x3) ≈ gr_cpu3(x3)
    @test gr_auto(x4) ≈ gr_cpu4(x4)
    @test gr_auto_c(x3) ≈ gr_cpu3c(x3)

    # GR CUDA auto path (view-based) matches explicit CUDA path
    @test DeconvOptim.GR_cuda(num_dims=nothing)(x1) ≈ DeconvOptim.GR_cuda(num_dims=1)(x1)
    @test DeconvOptim.GR_cuda(num_dims=nothing)(x2) ≈ DeconvOptim.GR_cuda(num_dims=2)(x2)
    @test DeconvOptim.GR_cuda(num_dims=nothing)(x3) ≈ DeconvOptim.GR_cuda(num_dims=3)(x3)
    @test DeconvOptim.GR_cuda(num_dims=nothing)(x4) ≈ DeconvOptim.GR_cuda(num_dims=4)(x4)
    @test DeconvOptim.GR_cuda()(x2) ≈ gr_auto(x2)
    @test DeconvOptim.GR_cuda()(x4) ≈ gr_auto(x4)

    # TV: num_dims=nothing selects the same explicit config on CPU
    tv_auto = TV()
    tv_auto_c = TV(mode="central")
    tv_cpu1 = TV(num_dims=1, weights=[1])
    tv_cpu2 = TV(num_dims=2, weights=[1, 1])
    tv_cpu3 = TV(num_dims=3, weights=[1, 1, 1])
    tv_cpu3c = TV(num_dims=3, weights=[1, 1, 1], mode="central")
    @test tv_auto(x1) ≈ tv_cpu1(x1)
    @test tv_auto(x2) ≈ tv_cpu2(x2)
    @test tv_auto(x3) ≈ tv_cpu3(x3)
    @test tv_auto_c(x3) ≈ tv_cpu3c(x3)

    # TV autoc PU == view kernel
    @test tv_auto(x2) ≈ DeconvOptim.TV_cuda(num_dims=nothing)(x2)
    @test tv_auto(x3) ≈ DeconvOptim.TV_cuda(num_dims=nothing)(x3)

    # TH: num_dims=nothing selects 1/2/3-D kernel automatically
    th_auto = TH()
    th_cpu1 = TH(num_dims=1)
    th_cpu2 = TH(num_dims=2)
    th_cpu3 = TH(num_dims=3)
    @test th_auto(x1) ≈ th_cpu1(x1)
    @test th_auto(x2) ≈ th_cpu2(x2)
    @test th_auto(x3) ≈ th_cpu3(x3)

    # TH CUDA auto path via TH_view
    th_cuda_auto = DeconvOptim.TH_cuda(num_dims=nothing)
    th_cuda1 = DeconvOptim.TH_cuda(num_dims=1)
    th_cuda2 = DeconvOptim.TH_cuda(num_dims=2)
    th_cuda3 = DeconvOptim.TH_cuda(num_dims=3)
    @test th_cuda_auto(x1) ≈ th_cuda1(x1)
    @test th_cuda_auto(x2) ≈ th_cuda2(x2)
    @test th_cuda_auto(x3) ≈ th_cuda3(x3)
    @test th_cuda_auto(x2) ≈ th_auto(x2)
    th_auto_view1 = DeconvOptim.TH_view(x1)
    th_auto_view2 = DeconvOptim.TH_view(x2)
    th_auto_view3 = DeconvOptim.TH_view(x3)
    @test th_auto_view1 ≈ th_cpu1(x1)
    @test th_auto_view2 ≈ th_cpu2(x2)
    @test th_auto_view3 ≈ th_cpu3(x3)

    # TH weights: CPU (@tullio) == CUDA (view) and explicit == what weights do
    thw_cpu = TH(num_dims=2, weights=[0.5, 2.0])
    thw_cuda = DeconvOptim.TH_cuda(num_dims=2, weights=[0.5, 2.0])
    @test thw_cuda(x2) ≈ thw_cpu(x2)
    thw3_cpu = TH(num_dims=3, weights=[1.0, 2.0, 3.0])
    thw3_cuda = DeconvOptim.TH_cuda(num_dims=3, weights=[1.0, 2.0, 3.0])
    @test thw3_cuda(x3) ≈ thw3_cpu(x3)

    # Tikhonov: num_dims=nothing selects the same explicit config on CPU
    tk_auto = Tikhonov()
    tk_auto_c = Tikhonov(mode="spatial_grad_square", step=2)
    tk_cpu1 = Tikhonov(num_dims=1, weights=[1])
    tk_cpu2 = Tikhonov(num_dims=2, weights=[1, 1])
    tk_cpu3 = Tikhonov(num_dims=3, weights=[1, 1, 1])
    tk_cpu3g = Tikhonov(num_dims=3, weights=[1, 1, 1], mode="spatial_grad_square", step=2)
    @test tk_auto(x1) ≈ tk_cpu1(x1)
    @test tk_auto(x2) ≈ tk_cpu2(x2)
    @test tk_auto(x3) ≈ tk_cpu3(x3)
    @test tk_auto_c(x3) ≈ tk_cpu3g(x3)

    # Tikhonov CPU auto == CUDA (view) auto path
    @test tk_auto(x2) ≈ DeconvOptim.Tikhonov_cuda(num_dims=nothing)(x2)
    @test tk_auto(x3) ≈ DeconvOptim.Tikhonov_cuda(num_dims=nothing)(x3)
    @test DeconvOptim.Tikhonov_cuda(num_dims=nothing)(x1) ≈ DeconvOptim.Tikhonov_cuda(num_dims=1)(x1)
    @test DeconvOptim.Tikhonov_cuda(num_dims=nothing)(x4) ≈ DeconvOptim.Tikhonov_cuda(num_dims=4)(x4)

    # unsupported cases throw
    @test_throws ArgumentError GR(mode="bogus")
    @test_throws ArgumentError DeconvOptim.GR_cuda(mode="bogus")
    @test_throws ArgumentError Tikhonov(mode="bogus")
    @test_throws ArgumentError DeconvOptim.Tikhonov_cuda(mode="bogus")
    @test_throws ArgumentError th_auto(x4)
    @test_throws ArgumentError th_cuda_auto(x4)
    @test_throws ArgumentError DeconvOptim.TH_view(x4)
end

@testset "TH sum_dims" begin
    x3 = abs.(randn(Float64, (6, 6, 6)))

    # sum_dims requiring all dimensions equals the full regularizer
    th_full = TH(num_dims=3)
    th_sum123 = TH(num_dims=3, sum_dims=[1, 2, 3])
    @test th_sum123(x3) ≈ th_full(x3)

    # sum_dims with fewer dims == sum over the per-slice regularizer
    th_sum12_2d = TH(num_dims=2, sum_dims=[1, 2])
    th_sum12_3d = TH(num_dims=3, sum_dims=[1, 2])
    @test th_sum12_3d(x3) ≈ sum(k -> th_sum12_2d(view(x3, :, :, k)), axes(x3, 3))

    # sum_dims=[1,3] == sum over the per-slice regularizer on slices (:, j, :)
    th13_2d = TH(num_dims=2, sum_dims=[1, 2])
    th13_3d = TH(num_dims=3, sum_dims=[1, 3])
    @test th13_3d(x3) ≈ sum(j -> th13_2d(view(x3, :, j, :)), axes(x3, 2))

    # weights positional: weights[k] pairs with s_dims[k]
    thw = TH(num_dims=3, sum_dims=[3], weights=[2.0])
    rw = 0.0
    for k in 2:5, j in axes(x3, 2), i in axes(x3, 1)
        rw += sqrt(1f-8 + 4 * (x3[i, j, k+1] + x3[i, j, k-1] - 2 * x3[i, j, k])^2)
    end
    @test thw(x3) ≈ rw

    # positional weights + cross terms: sum_dims=[1,2]
    thw2 = TH(num_dims=2, sum_dims=[1, 2], weights=[2.0, 3.0])
    x2 = abs.(randn(Float64, (6, 6)))
    rw2 = 0.0
    for j in 2:5, i in 2:5
        rw2 += sqrt(1f-8 +
            4 * (x2[i+1, j] + x2[i-1, j] - 2 * x2[i, j])^2 +
            9 * (x2[i, j+1] + x2[i, j-1] - 2 * x2[i, j])^2 +
            12 * (x2[i+1, j+1] - x2[i+1, j] - x2[i, j+1] + x2[i, j])^2)
    end
    @test thw2(x2) ≈ rw2

    # CUDA (view-based) path parity on CPU arrays
    th_cuda = DeconvOptim.TH_cuda(num_dims=3, sum_dims=[1, 2])
    @test th_cuda(x3) ≈ th_sum12_3d(x3)
    th_cuda_w = DeconvOptim.TH_cuda(num_dims=2, sum_dims=[1, 2], weights=[2.0, 3.0])
    @test th_cuda_w(x2) ≈ thw2(x2)

    # auto num_dims + sum_dims
    th_auto = TH(sum_dims=[1, 2])
    @test th_auto(x3) ≈ th_sum12_3d(x3)

    # out of range / bad sum_dims throw
    @test_throws ArgumentError TH(num_dims=2, sum_dims=[1, 3])(x3)
    @test_throws ArgumentError TH(sum_dims=[4])(x3)
    @test_throws ArgumentError DeconvOptim.TH_cuda(num_dims=2, sum_dims=[3])(x2)
end
