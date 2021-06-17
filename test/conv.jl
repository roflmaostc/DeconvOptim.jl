@testset "Convolution methods" begin

    conv_gen(u, v, dims) = real(ifft(fft(u, dims) .* fft(v, dims), dims))

    function conv_test(psf, img, img_out, dims, s)
        otf = fft(psf, dims)
        otf_r = rfft(psf, dims)
        otf_p, conv_p = plan_conv(img, psf, dims)
        otf_p2, conv_p2 = plan_conv(img .+ 0.0im, 0.0im .+ psf, dims)
        otf_p3, conv_p3 = plan_conv_psf(img, fftshift(psf,dims), dims)
        @testset "$s" begin
            @test img_out ≈ conv(0.0im .+ img, psf, dims)
            @test img_out ≈ conv(img, psf, dims)
            @test img_out ≈ conv_p(img, otf_p)
            @test img_out ≈ conv_p(img)
            @test img_out ≈ conv_p2(img .+ 0.0im, otf_p2)
            @test img_out ≈ conv_p2(img .+ 0.0im)
            @test img_out ≈ conv_psf(img, fftshift(psf, dims), dims)
            @test img_out ≈ conv_p3(img)
        end
    end
    

    N = 5
    psf = zeros((N, N))
    psf[1, 1] = 1
    img = randn((N, N))
    conv_test(psf, img, img, [1,2], "Convolution random image with delta peak")


    N = 5
    psf = zeros((N, N))
    psf[1, 1] = 1
    img = randn((N, N, N))
    conv_test(psf, img, img, [1,2], "Convolution with different dimensions psf, img delta")


    N = 5
    psf = abs.(randn((N, N, 2)))
    img = randn((N, N, 2))
    dims = [1, 2]
    img_out = conv_gen(img, psf, dims)
    conv_test(psf, img, img_out, dims, "Convolution with random 3D PSF and random 3D image over 2D dimensions")
 
    N = 5
    psf = abs.(randn((N, N, N, N, N)))
    img = randn((N, N, N, N, N))
    dims = [1, 2, 3, 4]
    img_out = conv_gen(img, psf, dims)
    conv_test(psf, img, img_out, dims, "Convolution with random 5D PSF and random 5D image over 4 Dimensions")

    N = 5
    psf = abs.(zeros((N, N, N, N, N)))
    for i = 1:N
        psf[1,1,1,1, i] = 1
    end
    img = randn((N, N, N, N, N))
    dims = [1, 2, 3, 4]
    img_out = conv_gen(img, psf, dims)
    conv_test(psf, img, img, dims, "Convolution with 5D delta peak and random 5D image over 4 Dimensions")

    

   @testset "Check types" begin
        N = 10
        img = randn(Float32, (N, N))
        psf = abs.(randn(Float32, (N, N)))
        dims = [1, 2] 
        @test typeof(conv_gen(img, psf, dims)) == typeof(conv(img, psf))
        @test typeof(conv_gen(img, psf, dims)) != typeof(conv(img .+ 0f0im, psf))
        @test conv_gen(img, psf, dims) .+ 1f0im ≈ 1f0im .+ conv(img .+ 0f0im, psf)
    end


    @testset "Check type get_plan" begin
        @test plan_rfft === DeconvOptim.get_plan(typeof(1f0))  
        @test plan_fft === DeconvOptim.get_plan(typeof(1im))  
    end

    @testset "dims argument nothing" begin
        N = 5
        psf = abs.(randn((N, N, N, N, N)))
        img = randn((N, N, N, N, N))
        dims = [1,2,3,4,5] 
        @test conv(psf, img) ≈ conv(img, psf, dims)
        @test conv(psf, img) ≈ conv(psf, img, dims)
        @test conv(img, psf) ≈ conv(img, psf, dims)
    end

    @testset "adjoint convolution" begin
        x = randn(ComplexF32, (5,6))
        y = randn(ComplexF32, (5,6))

        y_ft, p = plan_conv(x, y)
        @test ≈(exp(1im * 1.23) .+ conv(ones(eltype(y), size(x)), conj.(y)), exp(1im * 1.23) .+ Zygote.gradient(x -> sum(real(conv(x, y))), x)[1], rtol=1e-4)   
        @test ≈(exp(1im * 1.23) .+ conv(ones(ComplexF32, size(x)), conj.(y)), exp(1im * 1.23) .+ Zygote.gradient(x -> sum(real(p(x))), x)[1], rtol=1e-4) 
    end


end
