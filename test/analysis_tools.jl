

@testset "Analysis tools" begin
    # minimalistic test
    @testset "Relative Energy regain" begin
        x = [1f0 -2; 4 5; 7 8 ]
        @test DeconvOptim.relative_energy_regain(x, x .* 1) == (Float32[0.0, 0.333, 0.5, 0.601], Float32[1.0, 1.0, 1.0, 1.0])
        @test DeconvOptim.relative_energy_regain(x, x .* 0.5) == (Float32[0.0, 0.333, 0.5, 0.601], Float32[0.75, 0.75, 0.75, 0.75])
    end

    # minimalistic test
    @testset "Normalized Cross Correlation" begin
        x = [1f0 -2; 4 5; 7 8 ]
        
        @test DeconvOptim.normalized_cross_correlation(x, x) == 1.0f0
    end

    @testset "Trace Deconvolution" begin
        sz = (10,10)
        x = rand(sz...)
        psf = rand(10,10)
        psf /= sum(psf)
        y = DeconvOptim.conv(x,psf)
        # test whether starting with the ground truth really yield the perfect reconstruction after 0 iterations
        opt_options, summary = options_trace_deconv(x, 0, nothing)
        res = deconvolution(y,psf; initial=x, mapping=nothing, padding=0.0, opt_options=opt_options)

        @test summary["nccs"][1] > 0.999
        @test summary["nvars"][1] < 0.001
        @test (summary["best_nvar_img"] ≈ x)
    end
end
