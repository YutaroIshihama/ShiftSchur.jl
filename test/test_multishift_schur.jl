using Test
using LinearAlgebra
using Random
using ShiftSchur

# The multi-shift implementation exists in the repository but is not yet part
# of src/ShiftSchur.jl. Load the implementation files explicitly for this
# focused local regression test. The modules avoid name collisions with the
# already-exported ShiftSchur.schur_matrix.
module AuxMSLocal
using LinearAlgebra
include(joinpath(@__DIR__, "..", "src", "operators", "auxiliary_linearization.jl"))
include(joinpath(@__DIR__, "..", "src", "solvers", "multishift_arnoldi.jl"))
include(joinpath(@__DIR__, "..", "src", "solvers", "schur_multishift.jl"))
end
const AMS = AuxMSLocal

Random.seed!(42)

@testset "Schur + one-basis multi-shift" begin
    n, m = 10, 6
    A = randn(ComplexF64, n, n)
    A = A + A'
    D = Diagonal(collect(2.0:0.4:4.0))
    B = 0.2 .* randn(ComplexF64, n, m)
    b = randn(ComplexF64, n)
    zs = ComplexF64[-0.7 + 0.08im, 0.1 + 0.08im, 0.9 + 0.08im, 1.7 + 0.08im]

    X, info = AMS.schur_multishift(A, B, D, b, zs; krylovdim=n+m, tol=1e-11)

    @test length(X) == length(zs)
    @test length(info.residuals) == length(zs)
    @test info.iterations <= n + m

    for (x, z) in zip(X, zs)
        xref = AMS.schur_matrix(A, B, D, z) \ b
        @test x ≈ xref rtol=2e-9 atol=2e-10
        @test norm(AMS.schur_matrix(A, B, D, z) * x - b) / norm(b) < 2e-9
    end
end
