using Random
using ShiftSchur

include("../common/kubo.jl")

rng = MersenneTwister(20241033)
M = MetalCommon.synthetic_metal_model(m=1000, n=120, metal=:Fe2MnAl).schur_model
b = randn(rng, ComplexF64, size(M.A, 1))
ωgrid = range(-1.5, 1.5; length=121)
ηs = [0.2, 0.1, 0.05, 0.02]
rows = NamedTuple[]

println("=== Fe2MnAl a posteriori adaptive shifts ===")
for η in ηs
    comparison = adaptive_choice_vs_default_comparison(
        M, ωgrid, η, b; tol=1e-8, s_max=16)
    adaptive = comparison.adaptive
    default = comparison.default
    push!(rows, (η=η, default_k=default.k, default_max_err=default.max_err,
                 adaptive_k=adaptive.k, adaptive_max_err=adaptive.max_err))
    println("η=$η default: k=$(default.k), max_err=$(default.max_err); " *
            "adaptive: k=$(adaptive.k), max_err=$(adaptive.max_err)")
end

output = joinpath(@__DIR__, "../../docs/Fe2MnAl_adaptive_vs_default.csv")
open(output, "w") do io
    println(io, "eta,default_k,default_max_err,adaptive_k,adaptive_max_err")
    for row in rows
        println(io, "$(row.η),$(row.default_k),$(row.default_max_err)," *
                   "$(row.adaptive_k),$(row.adaptive_max_err)")
    end
end
println("Saved comparison to $output")
