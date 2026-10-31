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

@testset "Multi-shift scaling with number of shifts" begin
    n, m = 10, 6

    A = randn(ComplexF64, n, n)
    A = A + A'

    D = Diagonal(collect(2.0:0.4:4.0))
    B = 0.2 .* randn(ComplexF64, n, m)
    b = randn(ComplexF64, n)

    η = 0.08
    krylovdim = n + m

    nshifts_list = [1, 4, 41, 401]

    println()
    println("=== Multi-shift scaling test ===")
    println("naux      = ", n + m)
    println("krylovdim = ", krylovdim)
    println("eta       = ", η)
    println()

    @printf(
        "%10s  %10s  %12s  %14s  %14s  %12s\n",
        "nshifts",
        "iterations",
        "H-matvecs",
        "max_relerr",
        "max_resid",
        "time_sec",
    )

    reference_iterations = nothing

    for nshifts in nshifts_list
        ωs = collect(range(-1.0, 1.8; length=nshifts))
        zs = ComplexF64.(ωs .+ im * η)

        elapsed = @elapsed begin
            X, info = AMS.schur_multishift(
                A,
                B,
                D,
                b,
                zs;
                krylovdim=krylovdim,
                tol=1e-11,
            )
        end

        # In the current unrestarted Arnoldi implementation,
        # one H application is performed per Arnoldi iteration.
        matvecs = info.iterations

        errors = Float64[]
        residuals = Float64[]

        for (x, z) in zip(X, zs)
            S = AMS.schur_matrix(A, B, D, z)
            xref = S \ b

            relerr =
                norm(x - xref) /
                max(norm(xref), eps(Float64))

            residual =
                norm(S * x - b) /
                norm(b)

            push!(errors, relerr)
            push!(residuals, residual)
        end

        maxerr = maximum(errors)
        maxres = maximum(residuals)

        @printf(
            "%10d  %10d  %12d  %14.6e  %14.6e  %12.6f\n",
            nshifts,
            info.iterations,
            matvecs,
            maxerr,
            maxres,
            elapsed,
        )

        @test length(X) == nshifts
        @test length(info.residuals) == nshifts
        @test info.iterations <= krylovdim

        @test maxerr < 2e-9
        @test maxres < 2e-9

        if reference_iterations === nothing
            reference_iterations = info.iterations
        else
            # This is the important multi-shift property:
            # increasing the number of frequencies must NOT
            # increase the expensive large-operator Arnoldi work.
            @test info.iterations == reference_iterations
        end
    end

    println()
    println("Expected scaling:")
    println("  number of shifts        = 1 -> 4 -> 41 -> 401")
    println("  large Haux matvec count = constant")
    println("  only small projected shifted solves increase with nshifts")
end
