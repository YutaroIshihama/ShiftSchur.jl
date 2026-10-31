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

@testset "Multi-shift Krylov-dimension convergence" begin
    n, m = 10, 6

    A = randn(ComplexF64, n, n)
    A = A + A'

    D = Diagonal(collect(2.0:0.4:4.0))
    B = 0.2 .* randn(ComplexF64, n, m)
    b = randn(ComplexF64, n)

    η = 0.08
    ωs = collect(range(-1.0, 1.8; length=41))
    zs = ComplexF64.(ωs .+ im * η)

    kdims = [4, 6, 8, 10, 12, 14, 16]

    # Direct Schur solutions:
    # compute these once and use them as the reference for all Krylov dimensions.
    Xref = [
        AMS.schur_matrix(A, B, D, z) \ b
        for z in zs
    ]

    println()
    println("=== Multi-shift Krylov-dimension convergence ===")
    println("naux = ", n + m)
    println("nshifts = ", length(zs))
    println("eta = ", η)
    println()
    @printf("%6s  %10s  %14s  %14s  %14s  %14s\n",
            "kdim", "iterations", "max_relerr", "mean_relerr",
            "max_resid", "max_projres")

    max_errors = Float64[]
    mean_errors = Float64[]
    max_residuals = Float64[]
    max_projected_residuals = Float64[]

    for kdim in kdims
        X, info = AMS.schur_multishift(
            A, B, D, b, zs;
            krylovdim=kdim,
            tol=1e-11,
        )

        errors = Float64[]
        residuals = Float64[]

        for (x, xref, z) in zip(X, Xref, zs)
            S = AMS.schur_matrix(A, B, D, z)

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
        meanerr = sum(errors) / length(errors)
        maxres = maximum(residuals)
        maxproj = maximum(info.residuals)

        push!(max_errors, maxerr)
        push!(mean_errors, meanerr)
        push!(max_residuals, maxres)
        push!(max_projected_residuals, maxproj)

        @printf("%6d  %10d  %14.6e  %14.6e  %14.6e  %14.6e\n",
                kdim,
                info.iterations,
                maxerr,
                meanerr,
                maxres,
                maxproj)

        @test info.iterations <= kdim
        @test length(X) == length(zs)
    end

    println()
    println("Convergence ratios (max error):")

    for i in 2:length(kdims)
        ratio = max_errors[i-1] / max_errors[i]
        @printf("k=%2d -> %2d : %12.4e\n",
                kdims[i-1], kdims[i], ratio)
    end

    println()
    println("Final full-space values:")
    println("max relative error     = ", max_errors[end])
    println("mean relative error    = ", mean_errors[end])
    println("max Schur residual     = ", max_residuals[end])
    println("max projected residual = ", max_projected_residuals[end])

    # The full auxiliary dimension is n+m=16.
    # At kdim=16 Arnoldi spans the complete space (up to roundoff),
    # so it should reproduce the direct Schur solutions.
    @test kdims[end] == n + m
    @test max_errors[end] < 2e-9
    @test max_residuals[end] < 2e-9

    # Basic sanity check: reduced spaces should ultimately improve strongly.
    @test max_errors[end] < max_errors[1]
end
