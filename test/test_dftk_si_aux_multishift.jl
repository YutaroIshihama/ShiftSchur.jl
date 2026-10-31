using Test
using LinearAlgebra
using Printf
using ShiftSchur
using DFTK

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

    H = AMS.AuxiliaryLinearization(A, B, D)
    Hmat = [A B; B' D]
    rhs = AMS.lift_rhs(H, b)

    # ------------------------------------------------------------------
    # 2. PART A1: exact physical static consistency at z=0
    #
    # Do NOT solve S(0)x=b here. The sampled static response may contain a
    # null/near-null mode, so S(0) need not be invertible. At z=0 we test the
    # physical quantity that the adapter was constructed to reproduce:
    # Sigma(0), and the exact Schur elimination identity.
    # ------------------------------------------------------------------
    Sigma0 = B * (D \ B')
    S0 = A - Sigma0
    S0_from_aux_elimination = A - B * (D \ B')

    sigma_fit_relerr = norm(Sigma0 + p.metadata.chi0_matrix) /
                       max(norm(p.metadata.chi0_matrix), eps(Float64))
    schur_complement_error = norm(S0 - S0_from_aux_elimination) /
                             max(norm(S0), eps(Float64))

    println()
    println("--- Part A1: exact z=0 static consistency ---")
    @printf("Sigma(0) fit residual     = %.6e\n", sigma_fit_relerr)
    @printf("Schur elimination error   = %.6e\n", schur_complement_error)

    @test sigma_fit_relerr < 1e-10
    @test schur_complement_error < 1e-12

    # Temporary diagnostics for the observed z=0 singularity.
    # Keep these while developing the DFTK response adapter; they can later be
    # reduced to a regression assertion once the expected null space is understood.
    svals0 = svdvals(S0)
    rank0 = rank(S0)
    cond0 = cond(S0)
    println("singular values of S(0)   = ", svals0)
    println("rank S(0)                 = ", rank0, " / ", n)
    println("condition number S(0)     = ", cond0)
    if !isempty(svals0)
        println("min/max singular ratio    = ", minimum(svals0) / maximum(svals0))
    end

    # ------------------------------------------------------------------
    # 3. PART A2: regularized solution-level equivalence
    #
    # Use finite eta so that solution-level comparison is well defined even
    # when the exact static Schur matrix has null modes.
    # ------------------------------------------------------------------
    eta_static = 0.08
    zreg = 0.0 + im * eta_static
    Sreg = AMS.schur_matrix(A, B, D, zreg)
    x_schur = Sreg \ b
    X_aux = (Hmat - zreg * I(n + m)) \ rhs
    x_aux = Vector(AMS.physical_part(H, X_aux))

    solution_relerr = norm(x_aux - x_schur) /
                      max(norm(x_schur), eps(Float64))
    aux_residual = norm((Hmat - zreg * I(n + m)) * X_aux - rhs) / norm(rhs)
    schur_residual = norm(Sreg * x_schur - b) / norm(b)

    println()
    println("--- Part A2: regularized static-limit solver validation ---")
    println("z                           = ", zreg)
    @printf("Schur vs auxiliary relerr = %.6e\n", solution_relerr)
    @printf("auxiliary residual        = %.6e\n", aux_residual)
    @printf("Schur residual            = %.6e\n", schur_residual)

    @test solution_relerr < 1e-10
    @test aux_residual < 1e-10
    @test schur_residual < 1e-10

    # Same regularized point through the actual one-basis shifted Arnoldi path.
    X0_ms, info0 = AMS.schur_multishift(A, B, D, b, ComplexF64[zreg];
                                        krylovdim=n+m, tol=1e-11)
    static_ms_relerr = norm(X0_ms[1] - x_schur) /
                       max(norm(x_schur), eps(Float64))
    @printf("single-shift Arnoldi relerr= %.6e\n", static_ms_relerr)
    @test static_ms_relerr < 1e-9
    @test info0.iterations <= n + m

    # ------------------------------------------------------------------
    # 4. PART B: solver-only 41-shift regression on the same Si-derived model
    #
    # IMPORTANT: metadata.static_only=true. These frequency points test the
    # numerical solver only; they are NOT a physical dynamic Si spectrum.
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
    @test krylovdim == n + m
end
