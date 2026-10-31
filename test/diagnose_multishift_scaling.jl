using Test
using LinearAlgebra
using Random
using Printf
using ShiftSchur

module AuxMSLocal
using LinearAlgebra

include(joinpath(
    @__DIR__,
    "..",
    "src",
    "operators",
    "auxiliary_linearization.jl",
))

include(joinpath(
    @__DIR__,
    "..",
    "src",
    "solvers",
    "multishift_arnoldi.jl",
))

include(joinpath(
    @__DIR__,
    "..",
    "src",
    "solvers",
    "schur_multishift.jl",
))

end

const AMS = AuxMSLocal

Random.seed!(42)

@testset "Multi-shift scaling with number of shifts" begin

    # ------------------------------------------------------------
    # Synthetic Schur problem
    # ------------------------------------------------------------

    n, m = 10, 6

    A = randn(ComplexF64, n, n)
    A = A + A'

    D = Diagonal(collect(2.0:0.4:4.0))

    B = 0.2 .* randn(ComplexF64, n, m)

    b = randn(ComplexF64, n)

    # ------------------------------------------------------------
    # Test parameters
    # ------------------------------------------------------------

    η = 0.08

    # Full auxiliary-space dimension.
    # We deliberately use the complete space here because this test
    # concerns scaling with the number of shifts, not Krylov compression.
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

    # The first calculation establishes the Arnoldi iteration count.
    # All subsequent shift counts must require the same number of
    # large auxiliary-operator matvecs.
    reference_iterations = nothing

    for nshifts in nshifts_list

        # Julia 1.13 does not allow
        #
        #     range(-1.0, 1.8; length=1)
        #
        # because the two endpoints differ.
        #
        # For the one-shift case use the midpoint of the interval.
        ωs = if nshifts == 1
            [0.4]
        else
            collect(range(-1.0, 1.8; length=nshifts))
        end

        zs = ComplexF64.(ωs .+ im * η)

        # --------------------------------------------------------
        # One Arnoldi basis -> all requested shifts
        # --------------------------------------------------------

        local X
        local info

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

        # In the present unrestarted Arnoldi implementation,
        # there is one large H_aux application per Arnoldi iteration.
        matvecs = info.iterations

        # --------------------------------------------------------
        # Compare every multi-shift solution against direct Schur
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # Diagnostics
        # --------------------------------------------------------

        @printf(
            "%10d  %10d  %12d  %14.6e  %14.6e  %12.6f\n",
            nshifts,
            info.iterations,
            matvecs,
            maxerr,
            maxres,
            elapsed,
        )

        # --------------------------------------------------------
        # Correctness tests
        # --------------------------------------------------------

        @test length(X) == nshifts

        @test length(info.residuals) == nshifts

        @test info.iterations <= krylovdim

        @test maxerr < 2e-9

        @test maxres < 2e-9

        # --------------------------------------------------------
        # Core multi-shift scaling test
        # --------------------------------------------------------

        if reference_iterations === nothing
            reference_iterations = info.iterations
        else
            # Increasing the number of frequencies must not increase
            # construction of the large-operator Arnoldi basis.
            @test info.iterations == reference_iterations
        end
    end

    println()
    println("Expected multi-shift property:")
    println("  number of shifts        = 1 -> 4 -> 41 -> 401")
    println("  large Haux matvec count = constant")
    println("  only small projected shifted solves scale with nshifts")
    println()
end
