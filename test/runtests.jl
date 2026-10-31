using Test
using ShiftSchur
include("test_auxiliary_linearization.jl")
include("test_schur_identity.jl")
include("test_rational_reduction.jl")
include("test_dielectric.jl")
include("test_adaptive.jl")
include("diagnose_adaptive_convergence.jl")
@testset "Adaptive convergence diagnostics" begin
    results = run_adaptive_convergence_diagnostic()
    @test !isempty(results)
    @test all(row.max_true_error >= 0 for row in results)
    @test all(row.max_estimator >= 0 for row in results)
end
