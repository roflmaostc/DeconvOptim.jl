

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

end
