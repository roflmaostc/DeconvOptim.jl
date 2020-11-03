@testset "Good's roughness" begin
    Random.seed!(42)
    img = abs.(100 .* randn((50, 50)))

    reg = GR()
    reg2 = GR(weights=[2,2])


end
