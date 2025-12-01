

function center_test(x1, x2, x3, y1, y2, y3)
    arr1 = randn((x1, x2, x3))
    arr2 = zeros((y1, y2, y3)) 

    center_set!(arr2, arr1)
    arr3 = center_extract(arr2, (x1, x2, x3))
    @test arr1 ≈ arr3
end

 # test center set and center extract methods
@testset "center methods" begin
    center_test(4, 4, 4, 6,7,4)
    center_test(5, 4, 4, 7, 8, 4)
    center_test(5, 4, 4, 8, 8, 8)
    center_test(6, 4, 4, 7, 8, 8)


    @test 1 == center_pos(1)
    @test 2 == center_pos(2)
    @test 2 == center_pos(3)
    @test 3 == center_pos(4)
    @test 3 == center_pos(5)
    @test 513 == center_pos(1024)
end



@testset "interpolate methods" begin
    x = [12,2,2,1,2,4,3,1]
    y = [12, 7, 2, 2, 2, 1.5, 1, 1.5, 2, 3, 4, 3.5, 3, 2, 1]
    @test y ≈ my_interpolate(x, (15))
    
    x = collect(0:0.05:3)
    y = my_interpolate(x, (2 * size(x)[1]))
    @test mean(exp.(x)) ≈ mean(exp.(x))
    
    
    x = sin.(collect(0:0.001:3))
    y = my_interpolate(x, size(x)[1] * 3 + 2)
    y2 = my_interpolate(y, size(x)[1])
    @test isapprox(y2, x, rtol=1e-6)

    
    x = [12,2,2,1,2,4,3,1]
    y = [12, 7, 2, 2, 2, 1.5, 1, 1.5, 2, 3, 4, 3.5, 3, 2, 1]
    y_2d = reshape(y, 1, :)
    x_2d = my_interpolate(reshape(x, 1, :), size(y_2d))
    @test isapprox(x_2d, y_2d, rtol=1e-6)

     
end




@testset "Generate downsample" begin
    ds = generate_downsample(2, [1,2], 2)
    @test [2.5] ≈ @invokelatest ds([1 2; 3 4]) 
    
    ds = generate_downsample(2, [2], 2)
    @test [1.5; 3.5; 5.5; 7.5] ≈ @invokelatest ds([1 2; 3 4; 5 6; 7 8]) 
    
    ds = generate_downsample(2, [1], 2)
    @test [2.0 3.0; 6.0 7.0] ≈ @invokelatest ds([1 2; 3 4; 5 6; 7 8]) 
end


@testset "Generate PSF method" begin
    # large aperture is delta peak
    out = zeros((5, 5))
    out[1,1] = 1
    
    @test out ≈ generate_psf((5, 5), 100)

    # pinhole aperture
    out = ones((10, 10))
    out ./= sum(out)
    @test out ≈ generate_psf((10, 10), 0.01)
    
    # normalized
    @test 1 ≈ sum(generate_psf((100, 100), 10))
end



@testset "rr methods" begin
    out = [1.4142135623730951 1.0 1.4142135623730951; 1.0 0.0 1.0; 1.4142135623730951 1.0 1.4142135623730951]
    out ≈ DeconvOptim.rr_2D((3,3))
    @test [0] ≈ DeconvOptim.rr_2D((1, 1))
    @test [2,1,0,1,2] ≈ DeconvOptim.rr_2D((5, 1))
    @test [3,2,1,0,1,2] ≈ DeconvOptim.rr_2D((6, 1))

    out = [1.7320508075688772 1.4142135623730951; 1.4142135623730951 1.0; 1.4142135623730951 1.0; 1.0 0.0]
    out = reshape(out, (2,2,2))
    @test out ≈ DeconvOptim.rr_3D((2,2,2))

end


@testset "next fast fft size" begin
    @test DeconvOptim.next_fast_fft_size(23) == 24
    @test DeconvOptim.next_fast_fft_size((23, 46)) == (24, 48)
end

