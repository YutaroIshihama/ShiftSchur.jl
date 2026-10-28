@testset "greedy adaptive rational shifts" begin
    M, b = synthetic_rational_model()
    ωgrid = range(-1.0, 1.0; length = 41)
    η = 0.1
    shifts, hist = greedy_rational_shifts(M, ωgrid, η, b; s_max = 8,
                                          init_shifts = ComplexF64[0.0 + im * η])

    @test length(shifts) <= 9
    @test length(unique(shifts)) == length(shifts)
    @test length(hist) >= 2
    @test all(hist[i].k <= hist[i + 1].k for i in 1:(length(hist) - 1))
    @test hist[end].max_err < hist[1].max_err / 100
    @test hist[end].max_err < 1e-4
end
