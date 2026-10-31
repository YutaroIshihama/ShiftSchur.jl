@testset "greedy adaptive rational shifts" begin
    M, b = synthetic_rational_model()
    ωgrid = range(-1.0, 1.0; length = 41)
    η = 0.1
    shifts, hist = greedy_rational_shifts_aposteriori(
        M, ωgrid, η, b; s_max = 15, tol = 1e-4,
        init_shifts = ComplexF64[0.0 + im * η])
    Q = rational_krylov_basis(M.D, M.B', shifts)
    RM = ReducedSchurModel(M, Q)
    max_true_error = maximum(ωgrid) do ω
        z = ω + im * η
        exact = apply_self_energy(M, z, b)
        approx = apply_reduced_self_energy(RM, z, b)
        norm(exact - approx) / norm(exact)
    end

    @test length(shifts) <= 16
    @test length(unique(shifts)) == length(shifts)
    @test length(hist) >= 2
    @test all(hist[i].k <= hist[i + 1].k for i in 1:(length(hist) - 1))
    @test hist[end].max_err < hist[1].max_err / 100
    @test hist[end].max_err < 1e-4
    @test max_true_error < 1e-6
end
