using ShiftSchur
using LinearAlgebra
using Random
using Plots

# testから synthetic model の定義を再利用
# (testファイルの関数をそのままコピー)
function synthetic_rational_model(; m = 150, n = 3, seed = 20240601)
    rng = MersenneTwister(seed)
    nb = m - 4
    bulk = [range(-3.0, -1.3; length = nb ÷ 2); range(1.3, 3.0; length = nb - nb ÷ 2)]
    λ = sort([bulk; -0.6; -0.1; 0.35; 0.8])
    U = Matrix(qr(randn(rng, ComplexF64, m, m)).Q)
    D = U * Diagonal(λ) * U'
    D = (D + D') / 2
    A = (X = randn(rng, ComplexF64, n, n); (X + X') / 2)
    w = [abs(x) < 1.1 ? 0.5 : 3 / sqrt(m) for x in λ]
    B = randn(rng, ComplexF64, n, m) * Diagonal(w) * U'
    b = randn(rng, ComplexF64, n)
    return SchurModel(A, D, B), b
end

M, b = synthetic_rational_model()
ωgrid = range(-1.0, 1.0; length=41)
ηs = [0.2, 0.1, 0.05, 0.02]
svals = (2,4,8,12,16)
shift_sets(s, η) = collect(range(-1.0, 1.0; length=s) .+ im*η)

# データ収集
all_rows = []
for η in ηs
    rows = rational_compression_diagnostics(M, ωgrid, η, b, [shift_sets(s, η) for s in svals])
    append!(all_rows, rows)
end

# Figure 1: ηごとの収束 (x=k, y=max Σerr, logスケール)
# SISC で一番見せたい図
plt1 = plot(yscale=:log10, xlabel="k = dim(Q)", ylabel="max_z ||Σ(z)-Σ_k(z)||/||Σ(z)||",
            title="Rational Krylov convergence (ω∈[-1,1])", legend=:bottomleft, dpi=300)

for η in ηs
    sub = filter(r->r.η==η && r.kind==:rational, all_rows)
    sort!(sub, by=r->r.k)
    plot!(plt1, [r.k for r in sub], [r.Σerr_max for r in sub], 
          marker=:circle, label="η=$η")
end
savefig(plt1, "examples/convergence_eta.png")
println("saved examples/convergence_eta.png")

# Figure 2: Rational vs Polynomial (η=0.1固定)
poly_rows = []
for η in [0.1]
    append!(poly_rows, rational_compression_diagnostics(M, ωgrid, η, b, collect(svals); kind=:polynomial))
end

plt2 = plot(yscale=:log10, xlabel="k", ylabel="max Σerr", 
            title="Rational vs Polynomial (η=0.1)", legend=:topright, dpi=300)
r_sub = filter(r->r.η==0.1, all_rows)
sort!(r_sub, by=r->r.k)
plot!(plt2, [r.k for r in r_sub], [r.Σerr_max for r in r_sub], marker=:o, label="rational")

sort!(poly_rows, by=r->r.k)
plot!(plt2, [r.k for r in poly_rows], [r.Σerr_max for r in poly_rows], marker=:square, label="polynomial")
savefig(plt2, "examples/rational_vs_poly.png")
println("saved examples/rational_vs_poly.png")

# Figure 3: Greedy adaptive (新規)
if isdefined(ShiftSchur, :greedy_rational_shifts)
    shifts, hist = greedy_rational_shifts(M, ωgrid, 0.1, b; s_max=16, init_shifts=ComplexF64[0.0+0.1im])
    plt3 = plot(yscale=:log10, xlabel="iteration (greedy)", ylabel="max err Σ*b",
                title="Greedy adaptive poles", marker=:diamond, legend=false, dpi=300)
    plot!(plt3, 1:length(hist), [h.max_err for h in hist])
    savefig(plt3, "examples/greedy.png")
    println("saved examples/greedy.png, shifts=$(length(shifts))")
end

println("All done. Check examples/*.png")