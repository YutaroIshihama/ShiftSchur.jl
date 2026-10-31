using Test, LinearAlgebra, Random

module AuxiliaryLinearizationTestSupport
using LinearAlgebra
include(joinpath(@__DIR__, "..", "src", "operators", "auxiliary_linearization.jl"))
end

Random.seed!(20261031)
const Aux = AuxiliaryLinearizationTestSupport

@testset "auxiliary linearization agrees with the frequency-dependent Schur system" begin
    n, m = 7, 4
    Q = randn(ComplexF64, n, n)
    A = Q + Q' + 4I
    B = 0.15 .* randn(ComplexF64, n, m)
    D = Diagonal([-1.4, 0.2, 1.1, 2.3])
    b = randn(ComplexF64, n)
    zs = ComplexF64[-0.8 + 0.08im, 0.185 + 0.015im, 0.9 + 0.06im,
                    2.27 + 0.025im, 3.2 + 0.12im]

    H = Aux.AuxiliaryLinearization(A, B, D)
    Hmatrix = [A B; B' Matrix(D)]
    rhs = Aux.lift_rhs(H, b)

    for z in zs
        # Solve the shifted fixed auxiliary system, then compare its physical block.
        X = (Hmatrix - z * I(n + m)) \ rhs
        schur = Aux.schur_matrix(A, B, D, z)
        x = schur \ b

        relative_error = norm(Aux.physical_part(H, X) - x) / norm(x)
        auxiliary_residual = norm(H * X - z * X - rhs) / norm(rhs)
        schur_residual = norm(schur * x - b) / norm(b)

        @test relative_error < 2e-10
        @test auxiliary_residual < 2e-11
        @test schur_residual < 2e-11
    end
end

@testset "auxiliary frequency dependence is only the scalar shift" begin
    n, m = 7, 4
    Q = randn(ComplexF64, n, n)
    A = Q + Q' + 4I
    B = 0.15 .* randn(ComplexF64, n, m)
    D = Diagonal([-1.4, 0.2, 1.1, 2.3])
    H = Aux.AuxiliaryLinearization(A, B, D)
    Hmatrix = [A B; B' Matrix(D)]
    v = randn(ComplexF64, n + m)

    # Construct and apply the auxiliary operator once; each shift only subtracts z*v.
    Hv = H * v
    @test Hv ≈ Hmatrix * v
    for z in ComplexF64[-0.8 + 0.08im, 0.185 + 0.015im, 0.9 + 0.06im,
                        2.27 + 0.025im, 3.2 + 0.12im]
        shifted_action = Hv - z * v
        @test shifted_action ≈ Hmatrix * v - z * v
    end
end
