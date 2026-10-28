using LinearAlgebra, Random

@testset "Schur identity (random Hermitian)" begin
    rng = MersenneTwister(1234)
    n, m = 6, 14
    hermitian(k) = (X = randn(rng, ComplexF64, k, k); (X + X') / 2)
    A, D = hermitian(n), hermitian(m)
    B = randn(rng, ComplexF64, n, m) / 4
    b = randn(rng, ComplexF64, n)
    M = SchurModel(A, D, B)
    H = auxiliary_matrix(M)
    @test ishermitian(H)
    for z in (0.3 + 1e-2im, -1.5 + 0.1im, 2.0 + 1.0im, 40.0 + 0.5im, 0.0 + 1.0im)
        x = solve_schur(M, z, b)
        xf = solve_full_upper(M, z, b)
        @test norm(x - xf) / norm(x) < 1e-11
        @test norm(apply_schur(M, z, x) - b) / norm(b) < 1e-11
        v = randn(rng, ComplexF64, n)
        @test apply_schur(M, z, v) ≈ schur_matrix(M, z) * v
        @test apply_self_energy(M, z, v) ≈ self_energy(M, z) * v
    end
    # real z outside the spectrum
    zr = maximum(eigvals(Hermitian(H))) + 1.0
    @test norm(solve_schur(M, zr, b) - solve_full_upper(M, zr, b)) < 1e-11 * norm(solve_schur(M, zr, b))
end
