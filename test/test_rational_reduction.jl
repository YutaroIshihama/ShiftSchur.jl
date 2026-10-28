using LinearAlgebra, Random

# Synthetic model: D (m=150) has a gapped bulk spectrum in [-3,-1.3] ∪ [1.3,3]
# plus 4 discrete "resonance" levels inside the frequency interval [-1,1];
# B (n=3 × m) couples strongly to the resonances and weakly to the bulk.
# Random unitary eigenbasis, fixed seed => deterministic.
function synthetic_rational_model(; m = 150, n = 3, seed = 20240601)
    rng = MersenneTwister(seed)
    nb = m - 4
    bulk = [range(-3.0, -1.3; length = nb ÷ 2); range(1.3, 3.0; length = nb - nb ÷ 2)]
    λ = sort([bulk; -0.6; -0.1; 0.35; 0.8])
    U = Matrix(qr(randn(rng, ComplexF64, m, m)).Q)
    D = U * Diagonal(λ) * U'
    D = (D + D') / 2
    A = (X = randn(rng, ComplexF64, n, n); (X + X') / 2)
    w = [abs(x) < 1.1 ? 0.5 : 3 / sqrt(m) for x in λ]
    B = randn(rng, ComplexF64, n, m) * Diagonal(w) * U'
    b = randn(rng, ComplexF64, n)
    return SchurModel(A, D, B), b
end

@testset "Rational compression of Σ(z)" begin
    M, b = synthetic_rational_model()
    m = size(M.D, 1)
    ωmin, ωmax = -1.0, 1.0
    ωgrid = range(ωmin, ωmax; length = 41)
    ηs = (0.2, 0.1, 0.05, 0.02)
    shift_sets(s, η) = collect(range(ωmin, ωmax; length = s) .+ im * 0.1)
    svals = (2, 4, 8, 12, 16)

    @testset "basis properties" begin
        for s in svals
            Q = rational_krylov_basis(M.D, M.B', shift_sets(s, 0.1))
            RM = ReducedSchurModel(M, Q)
            @test size(Q, 2) <= 3 * (s + 1)
            @test size(Q, 2) >= 3 * (s + 1) - 3   # at most mild deflation
            @test norm(Q' * Q - I) < 1e-12
            @test norm(RM.T - RM.T') < 1e-12 * norm(RM.T)
            @test size(RM.T) == (size(Q, 2), size(Q, 2))
            @test size(RM.Bred) == (3, size(Q, 2))
            @test RM.Bred ≈ M.B * Q
        end
        # starting block contained in the space
        Q = rational_krylov_basis(M.D, M.B', shift_sets(3, 0.1))
        @test norm(M.B' - Q * (Q' * M.B')) < 1e-12 * norm(M.B')
    end

    @testset "rank-deficient / zero starting block" begin
        sh = shift_sets(3, 0.1)
        W = M.B'[:, [1, 1, 2]]                       # repeated column
        Q = rational_krylov_basis(M.D, W, sh)
        @test size(Q, 2) == 2 * 4
        @test norm(Q' * Q - I) < 1e-12
        Q0 = rational_krylov_basis(M.D, zeros(ComplexF64, m, 2), sh)
        @test size(Q0) == (m, 0)
        RM0 = ReducedSchurModel(M, Q0)
        @test norm(reduced_self_energy(RM0, 0.3 + 0.1im)) == 0
        @test solve_reduced_schur(RM0, 0.3 + 0.1im, b) ≈ (M.A - (0.3 + 0.1im) * I) \ b
        Q1 = rational_krylov_basis(M.D, M.B'[:, 1:1], sh)   # single vector start
        @test size(Q1, 2) == 4
        @test norm(Q1' * Q1 - I) < 1e-12
    end

    @testset "reduced linearization identity (x_aux == x_ROM, algebraic)" begin
        for s in (2, 6), z in (0.3 + 0.05im, -0.7 + 0.2im, 5.0 + 1.0im)
            RM = ReducedSchurModel(M, rational_krylov_basis(M.D, M.B', shift_sets(s, 0.1)))
            xr = solve_reduced_schur(RM, z, b)
            xa = solve_full_reduced_upper(RM, z, b)
            @test norm(xa - xr) / norm(xr) < 1e-11
            Haux = reduced_auxiliary_matrix(RM)
            @test norm(Haux - Haux') < 1e-12 * norm(Haux)
            v = b .+ 1
            @test apply_reduced_self_energy(RM, z, v) ≈ reduced_self_energy(RM, z) * v
        end
    end

    @testset "ROM approximation error over whole grid (x_ROM vs x_exact)" begin
        all_rows = NamedTuple[]
        for η in ηs
            rows = rational_compression_diagnostics(M, ωgrid, η, b, [shift_sets(s, η) for s in svals])
            append!(all_rows, rows)
            @test all(r.lin_err_max < 1e-9 for r in rows)
        end
        println("\n=== Rational Krylov (shifts ω_j + 0.1i), errors vs eta ===")
        print_diagnostics(all_rows)
        poly = NamedTuple[]
        for η in ηs
            append!(poly, rational_compression_diagnostics(M, ωgrid, η, b, collect(svals); kind = :polynomial))
        end
        println("\n=== Polynomial Krylov baseline ===")
        print_diagnostics(poly)

        r(η, s) = only(filter(x -> x.η == η && x.nshifts == s, all_rows))
        pr(kind, η, sv) = only(filter(x -> x.η == η && x.nshifts == sv, kind === :rational ? all_rows : poly))
        for η in ηs
            @test pr(:rational, η, 8).Σerr_max < 1e-3
            @test pr(:rational, η, 8).xerr_max < 1e-3
            @test pr(:rational, η, 16).Σerr_max < 1e-7
            @test pr(:rational, η, 16).xerr_max < 1e-7
            @test pr(:rational, η, 8).Σerr_max < pr(:rational, η, 4).Σerr_max / 100
            # rational beats polynomial Krylov at the same number of blocks
            @test pr(:rational, η, 8).Σerr_max < pr(:polynomial, η, 8).Σerr_max / 100
        end
        # smaller eta (closer to the real-axis poles) is harder at fixed small dimension
        @test pr(:rational, 0.02, 4).Σerr_max > pr(:rational, 0.2, 4).Σerr_max
    end
end
