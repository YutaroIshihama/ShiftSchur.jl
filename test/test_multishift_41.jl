using Test
using LinearAlgebra
using Random
using Printf
using ShiftSchur

module AuxMSLocal
using LinearAlgebra
include(joinpath(@__DIR__, "..", "src", "operators", "auxiliary_linearization.jl"))
include(joinpath(@__DIR__, "..", "src", "solvers", "multishift_arnoldi.jl"))
include(joinpath(@__DIR__, "..", "src", "solvers", "schur_multishift.jl"))
end

const AMS = AuxMSLocal

Random.seed!(42)

@testset "Schur + one-basis 41-shift test" begin
    n, m = 10, 6

    A = randn(ComplexF64, n, n)
    A = A + A'

    D = Diagonal(collect(2.0:0.4:4.0))
    B = 0.2 .* randn(ComplexF64, n, m)
    b = randn(ComplexF64, n)

    # 41 frequencies, all sharing the same imaginary broadening.
    η = 0.08
    ωs = collect(range(-1.0, 1.8; length=41))
    zs = ComplexF64.(ωs .+ im * η)

    # Full dimension here is intentional:
    # this test checks correctness of one-basis multi-shift first.
    krylovdim = n + m

    X, info = AMS.schur_multishift(
        A, B, D, b, zs;
        krylovdim=krylovdim,
        tol=1e-11,
    )

    @test length(X) == 41
    @test length(info.residuals) == 41
    @test info.iterations <= krylovdim

    relative_errors = Float64[]
    physical_residuals = Float64[]

    println()
    println("=== 41-shift Schur / auxiliary multi-shift test ===")
    println("nshifts      = ", length(zs))
    println("krylovdim    = ", krylovdim)
    println("iterations   = ", info.iterations)
    println()
    println("   omega       rel_error       Schur_residual     projected_residual")

    for (ω, z, x, projected_residual) in zip(ωs, zs, X, info.residuals)
        S = AMS.schur_matrix(A, B, D, z)
        xref = S \ b

        relerr = norm(x - xref) / max(norm(xref), eps(Float64))
        residual = norm(S * x - b) / norm(b)

        push!(relative_errors, relerr)
        push!(physical_residuals, residual)

        @printf(
            "%9.4f   %12.4e   %14.4e   %18.4e\n",
            ω,
            relerr,
            residual,
            projected_residual,
        )

        @test relerr < 2e-9
        @test residual < 2e-9
    end

    println()
    println("max relative error     = ", maximum(relative_errors))
    println("max Schur residual     = ", maximum(physical_residuals))
    println("max projected residual = ", maximum(info.residuals))
    println("Arnoldi iterations     = ", info.iterations)
    println("number of shifts       = ", length(zs))

    @test maximum(relative_errors) < 2e-9
    @test maximum(physical_residuals) < 2e-9
end
