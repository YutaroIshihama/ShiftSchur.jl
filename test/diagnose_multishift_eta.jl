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

@testset "Multi-shift eta dependence" begin
    n, m = 10, 6

    A = randn(ComplexF64, n, n)
    A = A + A'

    D = Diagonal(collect(2.0:0.4:4.0))
    B = 0.2 .* randn(ComplexF64, n, m)
    b = randn(ComplexF64, n)

    ωs = collect(range(-1.0, 1.8; length=41))

    etas = [0.20, 0.10, 0.08, 0.05, 0.02, 0.01]
    kdims = [8, 12, 14, 16]

    println()
    println("=== Multi-shift eta dependence ===")
    println("naux    = ", n + m)
    println("nshifts = ", length(ωs))
    println()

    @printf(
        "%8s  %6s  %10s  %14s  %14s  %14s\n",
        "eta",
        "kdim",
        "iterations",
        "max_relerr",
        "mean_relerr",
        "max_resid",
    )

    results = NamedTuple[]

    for η in etas
        zs = ComplexF64.(ωs .+ im * η)

        Xref = [
            AMS.schur_matrix(A, B, D, z) \ b
            for z in zs
        ]

        for kdim in kdims
            X, info = AMS.schur_multishift(
                A,
                B,
                D,
                b,
                zs;
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

            push!(
                results,
                (
                    eta=η,
                    kdim=kdim,
                    iterations=info.iterations,
                    max_relerr=maxerr,
                    mean_relerr=meanerr,
                    max_residual=maxres,
                    max_projected_residual=maxproj,
                ),
            )

            @printf(
                "%8.3f  %6d  %10d  %14.6e  %14.6e  %14.6e\n",
                η,
                kdim,
                info.iterations,
                maxerr,
                meanerr,
                maxres,
            )

            @test length(X) == 41
            @test info.iterations <= kdim

            if kdim == n + m
                @test maxerr < 2e-9
                @test maxres < 2e-9
            end
        end

        println()
    end

    println()
    println("=== Summary: max relative error ===")
    @printf("%8s", "eta")

    for kdim in kdims
        @printf("   k=%-10d", kdim)
    end

    println()

    for η in etas
        @printf("%8.3f", η)

        for kdim in kdims
            r = only(
                filter(
                    x -> x.eta == η && x.kdim == kdim,
                    results,
                ),
            )

            @printf("   %10.3e", r.max_relerr)
        end

        println()
    end

    println()
    println("=== Full-space consistency ===")

    for η in etas
        r = only(
            filter(
                x -> x.eta == η && x.kdim == n + m,
                results,
            ),
        )

        @printf(
            "eta=%6.3f : maxerr=%10.3e  maxres=%10.3e\n",
            η,
            r.max_relerr,
            r.max_residual,
        )
    end
end
