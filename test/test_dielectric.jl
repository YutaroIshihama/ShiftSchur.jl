using Test
using LinearAlgebra
using ShiftSchur

@testset "dielectric toy" begin
    M, b = synthetic_rational_model()
    Q = rational_krylov_basis(M.D, M.B', ComplexF64[0+0.1im, -0.5+0.1im, 0.5+0.1im])
    RM = ReducedSchurModel(M, Q)
    z = 0.2 + 0.1im
    Vc = Matrix{ComplexF64}(I, 3, 3)
    χ = chi0_reduced(RM, z)
    ε = dielectric_matrix_reduced(RM, z, Vc)
    @test size(χ) == (3, 3)
    @test size(ε) == (3, 3)
    @test ε ≈ I - Vc * χ
    @test dielectric_matrix_reduced(RM, z) ≈ ε
    εM = macroscopic_epsilon(ε)
    @test εM isa Number

    # the 3-shift basis gives ~6e-3 relative error; a richer basis meets 1e-3
    χ_exact = -self_energy(M, z)
    @test norm(χ - χ_exact) / norm(χ_exact) < 1e-2
    shifts = collect(range(-1.0, 1.0; length = 8) .+ 0.1im)
    RM8 = ReducedSchurModel(M, rational_krylov_basis(M.D, M.B', shifts))
    @test norm(chi0_reduced(RM8, z) - χ_exact) / norm(χ_exact) < 1e-3
end

@testset "dftk bridge" begin
    A = [1.0 0.5; 0.5 2.0]
    B = [0.1 0.2 0.3; 0.4 0.5 0.6]
    D = Diagonal([3.0, 4.0, 5.0]) |> Matrix
    H = [A B; B' D]
    M = build_schur_model_from_blocks(H, [true, true, false, false, false],
                                      [false, false, true, true, true])
    @test M.A == A && M.B == B && M.D == D
    M2 = build_schur_model(A, B, D)
    @test M2.A == A && M2.B == B && M2.D == D
    @test_throws DimensionMismatch build_schur_model_from_blocks(H, [true, false], [false, true])
    @test_throws ArgumentError build_schur_model_from_blocks(H, trues(5), trues(5))
    p = DFTKOccVirtPartition(2, 3; kpoint = 1)
    @test p.n_occ == 2 && p.n_virt == 3 && p.kpoint == 1

    L = linear_operator_adapter(B)
    x = [1.0, 2.0, 3.0]
    y = [1.0, -1.0]
    @test L * x ≈ B * x
    @test L' * y ≈ B' * y
end
