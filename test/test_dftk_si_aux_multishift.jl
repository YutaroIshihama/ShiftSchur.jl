using Test
using LinearAlgebra
using Printf
using ShiftSchur

# Existing real DFTK-Si SCF and static-response adapter.
include(joinpath(@__DIR__, "..", "src", "dftk", "si_scf.jl"))
include(joinpath(@__DIR__, "..", "src", "dftk", "si_response_adapter.jl"))
using .ShiftSchurDFTKSiSCF
using .ShiftSchurDFTKSiResponse

# Until these files are formally included/exported by src/ShiftSchur.jl,
# load the auxiliary/multi-shift implementation locally for this integration test.
module AuxMSLocal
using LinearAlgebra
include(joinpath(@__DIR__, "..", "src", "operators", "auxiliary_linearization.jl"))
include(joinpath(@__DIR__, "..", "src", "solvers", "multishift_arnoldi.jl"))
include(joinpath(@__DIR__, "..", "src", "solvers", "schur_multishift.jl"))
end
const AMS = AuxMSLocal

@testset "DFTK-Si static Schur -> auxiliary -> multi-shift integration" begin
    # ------------------------------------------------------------------
    # 1. Real DFTK-Si SCF and sampled static chi0 -> SchurModel
    # ------------------------------------------------------------------
    cfg = SiSCFConfig()
    scfres = run_si_scf(cfg)
    payload = scf_payload(scfres)

    smoke = dftk_static_response_smoke(payload)
    @test norm(smoke.delta_rho0) > 0

    p = build_si_response_problem(payload; q=(0.0, 0.0, 0.0),
                                  perturbation=:density_head, nmodes=6)
    @test validate_response_problem(p)
    @test p.metadata.static_only === true
    @test p.metadata.static_fit_relative_error < 1e-10

    M, b = p.model, p.probe
    A, B, D = Matrix(M.A), Matrix(M.B), Matrix(M.D)
    n, m = size(A, 1), size(D, 1)

    println()
    println("=== DFTK-Si auxiliary/multi-shift integration ===")
    println("DFTK version               = ", pkgversion(DFTK))
    println("physical Schur dimension   = ", n)
    println("auxiliary D dimension      = ", m)
    println("enlarged Haux dimension    = ", n + m)
    println("static fit relative error  = ", p.metadata.static_fit_relative_error)
    println("NOTE: dynamic shifts below are solver-only regression on static_only=true model")

    # ------------------------------------------------------------------
    # 2. PART A: physically meaningful static z=0 equivalence
    # ------------------------------------------------------------------
    H = AMS.AuxiliaryLinearization(A, B, D)
    Hmat = [A B; B' D]
    rhs = AMS.lift_rhs(H, b)

    z0 = 0.0 + 0.0im
    S0 = AMS.schur_matrix(A, B, D, z0)
    x_schur0 = S0 \ b
    X_aux0 = Hmat \ rhs
    x_aux0 = Vector(AMS.physical_part(H, X_aux0))

    static_relerr = norm(x_aux0 - x_schur0) / max(norm(x_schur0), eps(Float64))
    static_auxres = norm(Hmat * X_aux0 - rhs) / norm(rhs)
    static_schurres = norm(S0 * x_schur0 - b) / norm(b)

    Sigma0 = B * (D \ B')
    sigma_fit_relerr = norm(Sigma0 + p.metadata.chi0_matrix) /
                       max(norm(p.metadata.chi0_matrix), eps(Float64))

    println()
    println("--- Part A: z=0 physical static validation ---")
    @printf("Schur vs auxiliary relerr = %.6e\n", static_relerr)
    @printf("auxiliary residual        = %.6e\n", static_auxres)
    @printf("Schur residual            = %.6e\n", static_schurres)
    @printf("Sigma(0) fit residual     = %.6e\n", sigma_fit_relerr)

    @test static_relerr < 1e-10
    @test static_auxres < 1e-10
    @test static_schurres < 1e-10
    @test sigma_fit_relerr < 1e-10

    # Also exercise the actual multi-shift path at the physical static point.
    X0_ms, info0 = AMS.schur_multishift(A, B, D, b, ComplexF64[z0];
                                         krylovdim=n+m, tol=1e-11)
    static_ms_relerr = norm(X0_ms[1] - x_schur0) /
                       max(norm(x_schur0), eps(Float64))
    @printf("single-shift Arnoldi relerr= %.6e\n", static_ms_relerr)
    @test static_ms_relerr < 1e-9
    @test info0.iterations <= n + m

    # ------------------------------------------------------------------
    # 3. PART B: solver-only 41-shift regression on the same Si model
    # ------------------------------------------------------------------
    eta = 0.08
    omegas = collect(range(-0.8, 0.8; length=41))
    zs = ComplexF64.(omegas .+ im * eta)
    krylovdim = n + m

    Xms, info = AMS.schur_multishift(A, B, D, b, zs;
                                     krylovdim=krylovdim, tol=1e-11)

    relerrs = Float64[]
    schur_residuals = Float64[]
    for (x, z) in zip(Xms, zs)
        S = AMS.schur_matrix(A, B, D, z)
        xref = S \ b
        push!(relerrs, norm(x - xref) / max(norm(xref), eps(Float64)))
        push!(schur_residuals, norm(S * x - b) / norm(b))
    end

    max_relerr = maximum(relerrs)
    max_schurres = maximum(schur_residuals)
    max_projres = maximum(info.residuals)

    println()
    println("--- Part B: Si-derived solver-only 41-shift regression ---")
    println("number of shifts           = ", length(zs))
    println("Arnoldi iterations         = ", info.iterations)
    println("Haux matvecs (current impl)= ", info.iterations)
    @printf("max relative error         = %.6e\n", max_relerr)
    @printf("max Schur residual         = %.6e\n", max_schurres)
    @printf("max projected residual     = %.6e\n", max_projres)
    println("WARNING: Part B is NOT a physical dynamic Si spectrum.")

    @test length(Xms) == 41
    @test length(info.residuals) == 41
    @test info.iterations <= krylovdim
    @test max_relerr < 2e-9
    @test max_schurres < 2e-9

    # In full auxiliary dimension, one shared Arnoldi basis should serve all 41 shifts.
    @test krylovdim == n + m
end
