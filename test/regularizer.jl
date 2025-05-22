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


end

@testset "Good's roughness" begin

    x = generate_GR(5, [1,2], [4, 5], 1, -1, debug=true)

    @test x == Any[:(res = -2.0 * arr[i1, i2, i3, i4, i5] * (4 * (arr[i1 + 1, i2, i3, i4, i5] + arr[i1 + -1, i2, i3, i4, i5]) + 5 * (arr[i1, i2 + 1, i3, i4, i5] + arr[i1, i2 + -1, i3, i4, i5]) + -18 * arr[i1, i2, i3, i4, i5]))]

    x = [1,2,3,1,3,1,12.0,2,2,3,2.0]
    reg = GR(num_dims=1, sum_dims=[1], weights=[1])
    @test 22.71233466779126 ≈ reg(x)


end


@testset "TV" begin
    x = [1,2,3,1,3,1,12.0,2,2,3,2.0]
    reg = TV(num_dims=1, sum_dims=[1], weights=[1])
    @test 31.00010002845424 ≈ reg(x)

    # tests would fail:
    # @test TV_cuda(num_dims=2)(x) ≈ reg(x)
    # @test TV_cuda(num_dims=3)(x) ≈ reg(x)

    x = generate_TV(4, [1,2], [5, 7], 1, -1, debug=true)
    @test x == Any[:(res = sqrt(5 * abs2(arr[i1 + 1, i2, i3, i4] - arr[i1 + -1, i2, i3, i4]) + 7 * abs2(arr[i1, i2 + 1, i3, i4] - arr[i1, i2 + -1, i3, i4]) + 1.0f-8))]


    

end
