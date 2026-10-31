
# metal/Al40/run_Al40.jl — Al 40原子相当のデモ
using LinearAlgebra, Random
include("../../src/operators/schur_self_energy.jl")
include("../../src/operators/schur_operator.jl")
include("../../src/operators/auxiliary_operator.jl")
include("../../src/reduction/rational_krylov.jl")
include("../../src/reduction/reduced_model.jl")
include("../../src/reduction/adaptive_poles.jl")
include("../../src/reduction/diagnostics.jl")
include("../../src/response/occupation.jl")
include("../../src/response/chi0.jl")
include("../../src/response/epsilon.jl")
include("../../src/response/adaptive_choice_extra_bands_dynamic.jl")
include("../common/kubo.jl")

rng = MersenneTwister(20241031)
M = MetalCommon.synthetic_metal_model(m=600, n=60, metal=:Al).schur_model
b = randn(rng, ComplexF64, size(M.A,1))

ωgrid = range(0.0, 10.0, length=101)
ηs = [0.2, 0.1, 0.05, 0.02]

println("=== Al40 dynamic dielectric benchmark ===")
for η in ηs
    println("\n-- η=$η adaptive --")
    res = adaptive_extra_bands_dynamic(M, ωgrid, η, b; tol=1e-8, s_max=12)
    println("adaptive: n_extra=$(res.n_extra) max_err=$(res.max_err)")
    comp = adaptive_choice_vs_default_comparison(M, ωgrid, η, b)
    println("default k=$(comp.default.k) max_err=$(comp.default.max_err) vs adaptive k=$(comp.adaptive.k) max_err=$(comp.adaptive.max_err)")
end

qG = [0.1, 0.5, 1.0]
ε_sos = [epsilon_sos(M, ω, 0.1, qG, b) for ω in ωgrid]
println("Done Al40")
