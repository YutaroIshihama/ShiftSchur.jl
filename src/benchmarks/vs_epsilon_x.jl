
"""
vs_epsilon_x.jl — 主敵は SoS = QE epsilon.x (Adler-Wiser)

Cances論文では shifted Sternheimer を比較相手にしたが、動的 ω+iη では
条件数 κ~(bandwidth+ω)/η で Sternheimer が自滅するため不適切。
動的での真の産業標準は sum-over-states = QE epsilon.x / BerkeleyGW / YAMBO の
Adler-Wiser式:

 χ0_GG'(q,ω) = 1/Ω Σ_{k,n,m} (f_nk - f_mk+q)/(ε_nk - ε_mk+q + ω + iη) M* M

本ファイルは:
1. SoSの k点/N_cond 収束の遅さ (金属で致命的)
2. ShiftSchur圧縮の k_dim での高速収束
を同じメトリクス Σerr, xerr で比較するハーネス。

既存関数:
- SchurModel, self_energy, reduced_self_energy, solve_schur, solve_reduced_schur
- rational_krylov_basis, ReducedSchurModel, rational_compression_diagnostics
- greedy_rational_shifts_aposteriori (docs/estimators版)
を再利用。

保存先: src/benchmarks/vs_epsilon_x.jl
"""

module VsEpsilonX

using LinearAlgebra
using Printf

export BenchRow, benchmark_k_convergence, benchmark_eta_dependence, benchmark_ncond_convergence
export benchmark_sos_vs_schur_full, save_benchmark_csv

"""
BenchRow — Cances Fig.2 対応の1行
"""
struct BenchRow
    method::Symbol        # :sos, :schur, :schur_aposteriori
    k_dim::Int            # 縮小基底次元 dim(Q) = 実質 extra bands
    eta::Float64
    nshifts::Int
    Sigma_err_max::Float64
    Sigma_err_mean::Float64
    x_err_max::Float64
    ncond::Int            # SoSの場合の非占有バンド数
end

# --- SoS baseline: QE epsilon.x 相当の素朴実装 ---
"""
chi0_sos_bruteforce(M, z, b; ncond=0)

M.D の固有値を非占有バンドとみなして Adler-Wiser 和を直接計算。
M は SchurModel(A,D,B) だが、ここでは D を対角化して固有値を使う簡略版。
実際は H(k) の固有分解に対応。

Σ(z) = B (D - zI)^{-1} B' を
固有分解 D = U Λ U' で Σ(z)= (B U) (Λ - zI)^{-1} (B U)' と見て、
N_cond までで打ち切った誤差を測ることで SoS のバンド収束を模擬。

ncond=0 なら全バンド。
"""
function chi0_sos_bruteforce(M, z::Number, b::AbstractVector; ncond::Int=0)
    # 正確には self_energy(M,z) * b と同じだが、N_cond打ち切りで近似
    D = Matrix(M.D)
    # 対角化は重いので、m<=2000想定。DFTでは別の経路が必要だがベンチ用
    F = eigen(Hermitian(D))
    Λ = F.values
    U = F.vectors
    BU = M.B * U  # n x m
    # (Λ - zI)^{-1} BU' b
    Bp_b = BU' * b  # m
    if ncond > 0 && ncond < length(Λ)
        # エネルギー昇順で低エネルギーから ncond 個だけ使う = バンド打ち切り
        idx = sortperm(Λ)
        # 低エネルギー側を優先 (金属では Fermi 近傍が重要)
        sel = idx[1:min(ncond, end)]
        Bp_b_sel = Bp_b[sel]
        BU_sel = BU[:, sel]
        inv = 1.0 ./ (Λ[sel] .- z)
        x = BU_sel * (inv .* Bp_b_sel)
    else
        inv = 1.0 ./ (Λ .- z)
        x = BU * (inv .* Bp_b)
    end
    return x  # = Σ(z) b の近似
end

"""
benchmark_k_convergence(M, ωgrid, η, b; k_dims, ncond_list)

k_dims: 縮小次元リスト (例: [10,20,40,80])
ncond_list: SoSのバンド打ち切りリスト (例: [20,50,100,200,全])

両者を同じ ωgrid, η で比較。
"""
function benchmark_k_convergence(M, ωgrid, η::Real, b;
                                 k_dims::AbstractVector{Int}=[10,20,40,80,120],
                                 ncond_list::AbstractVector{Int}=[20,50,100,200,400],
                                 rtol::Real=1e-10)
    rows = BenchRow[]
    zref_list = [ComplexF64(ω+im*η) for ω in ωgrid]
    # 参照解: exact Σ(z) b (全D)
    Σref_norms = Float64[]
    for z in zref_list
        push!(Σref_norms, norm(self_energy(M,z)*b))
    end

    # --- ShiftSchur圧縮 ---
    # shifts を ωgrid から貪欲に選ぶ (exact不要版があればそちらを使う)
    shifts = ComplexF64[]
    try
        shifts_hist = greedy_rational_shifts_aposteriori(M, ωgrid, η, b; s_max=16, rtol=rtol)
        shifts = shifts_hist[1] isa Vector ? shifts_hist[1] : shifts_hist[1][1]
    catch
        # fallback: 等間隔
        shifts = ComplexF64.(collect(range(minimum(ωgrid), maximum(ωgrid), length=8) .+ im*η))
    end

    for kd in k_dims
        # kd に近い次元になるように shifts 数を調整
        # 簡略: shifts を kd÷3 個使う
        ns = clamp(kd ÷ 3, 1, length(shifts))
        sh = shifts[1:ns]
        Q = rational_krylov_basis(M.D, M.B', sh; rtol=rtol)
        RM = ReducedSchurModel(M, Q)
        # k が要求より大きい場合はそのまま (deflationで小さくなることも)
        es = Float64[]; ex = Float64[]
        for (i,z) in enumerate(zref_list)
            Σ_exact = self_energy(M,z)
            Σ_k = reduced_self_energy(RM,z)
            push!(es, norm(Σ_exact - Σ_k)/max(norm(Σ_exact),1e-16))
            x_exact = solve_schur(M,z,b)
            x_k = solve_reduced_schur(RM,z,b)
            push!(ex, norm(x_exact - x_k)/max(norm(x_exact),1e-16))
        end
        push!(rows, BenchRow(:schur, size(Q,2), η, length(sh), maximum(es), sum(es)/length(es), maximum(ex), 0))
    end

    # --- SoS: N_cond 打ち切り ---
    for nc in ncond_list
        es = Float64[]; ex = Float64[]
        for z in zref_list
            Σ_exact_b = self_energy(M,z)*b
            Σ_nc_b = chi0_sos_bruteforce(M,z,b; ncond=nc)
            push!(es, norm(Σ_exact_b - Σ_nc_b)/max(norm(Σ_exact_b),1e-16))
            # x_err は SoS では直接定義できないので Σb 誤差を流用
            push!(ex, es[end])
        end
        push!(rows, BenchRow(:sos, nc, η, 0, maximum(es), sum(es)/length(es), maximum(ex), nc))
    end

    return rows
end

"""
benchmark_eta_dependence(M, ωgrid, ηs, b)

ηが小さいほど SoS と多項式Krylov は死ぬが、rationalは生き残ることを示す。
Cances論文 Fig.3 対応。
"""
function benchmark_eta_dependence(M, ωgrid, ηs, b; k_fixed::Int=60)
    all_rows = BenchRow[]
    for η in ηs
        rows = benchmark_k_convergence(M, ωgrid, η, b; k_dims=[k_fixed], ncond_list=[100,400])
        append!(all_rows, rows)
    end
    return all_rows
end

"""
benchmark_ncond_convergence — QE epsilon.x の N_cond 収束そのもの

ε(ω) の虚部 Im ε ∝ Σ の虚部 が N_cond でどう収束するかを見る。
Al40 では 200バンド、Fe2MnAl では 500バンド必要、という主張を出すため。
"""
function benchmark_ncond_convergence(M, ω::Real, η::Real, b; ncond_list=20:20:500)
    z = ComplexF64(ω+im*η)
    Σ_exact_b = self_energy(M,z)*b
    errs = [(nc=nc, err=norm(Σ_exact_b - chi0_sos_bruteforce(M,z,b; ncond=nc))/norm(Σ_exact_b)) for nc in ncond_list]
    return errs
end

"""
benchmark_sos_vs_schur_full — 1関数で全比較 (Al40, Fe2MnAl 両用)

Returns DataFrame的 Vector{BenchRow}
"""
function benchmark_sos_vs_schur_full(M, ωgrid, η::Real, b)
    k_dims = [10,20,40,60,80,120,160]
    ncond_list = [20,50,100,200,400,600]
    return benchmark_k_convergence(M, ωgrid, η, b; k_dims=k_dims, ncond_list=ncond_list)
end

function save_benchmark_csv(rows::Vector{BenchRow}, path::String)
    open(path, "w") do io
        println(io, "method,k_dim,eta,nshifts,Sigma_err_max,Sigma_err_mean,x_err_max,ncond")
        for r in rows
            @printf(io, "%s,%d,%.6g,%d,%.6e,%.6e,%.6e,%d\n",
                String(r.method), r.k_dim, r.eta, r.nshifts, r.Sigma_err_max, r.Sigma_err_mean, r.x_err_max, r.ncond)
        end
    end
end

end # module
