
# metal/Fe2MnAl/run_Fe2MnAl.jl — Fe2MnAl Heusler, 難しい金属例
# Cances論文 Section 5.2, metals/Fe2MnAl 対応の動的版
# Al40より d-d 混成で共鳴が密、adaptive extra bands が効く例

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

rng = MersenneTwister(20241033)

# Fe2MnAl: m 大きめ、共鳴 6個、半金属的ギャップ近傍
M = MetalCommon.synthetic_metal_model(m=1000, n=120, metal=:Fe2MnAl).schur_model
b = randn(rng, ComplexF64, size(M.A,1))

ωgrid = range(-3.0, 3.0, length=121) # Fermi近傍重点: Heuslerは±2eVにdピーク
ηs = [0.1, 0.05, 0.02]

println("=== Fe2MnAl dynamic dielectric (Heusler hard case) ===")

# 1. Adaptive extra bands: 静的版 run_adaptive_algorithm.jl 対応の動的版
for η in ηs
    println("\n-- adaptive choice η=$η --")
    # 静的では extra_bands = f(gap, σ) だが、動的では ω依存
    result = adaptive_extra_bands_dynamic(M, ωgrid, η, b; tol=1e-8)
    println("adaptive result: n_extra=$(result.n_extra) shifts=$(length(result.shifts))")
    println("k_dim=$(result.k_dim) max_err=$(result.max_err)")

    # 2. 圧縮診断
    rows = rational_compression_diagnostics(M, ωgrid, η, b, [result.shifts]; kind=:rational)
    println(rows)
end

# 3. ε(ω) 比較: SoS vs Schur (Al40と同形式)
qG = [0.1, 0.5, 1.0]
println("Computing ε(ω) for Fe2MnAl...")
ε_sos_01 = [epsilon_sos(M, ω, 0.1, qG, b) for ω in ωgrid]
println("Done Fe2MnAl")
