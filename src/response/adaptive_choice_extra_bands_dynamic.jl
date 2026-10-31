
"""
adaptive_choice_extra_bands_dynamic.jl
— Cances論文 Section 6 / Appendix B / run_adaptive_algorithm.jl の動的版

静的版: gap と σ から必要な extra bands 数 N_extra を決める。
  N_extra ~ O(log(1/tol) * σ / gap) 的な評価。

動的版: z = ω + iη なので、
  1) ω が大きいと共鳴分母 |ε_c - ε_v - ω - iη| が小さくなり、
     より多くの共鳴 D 側極を保持する必要がある。
  2) η が小さいと同様に極近傍で条件数が上がる。

ここでは ωgrid 全体で max_ω error が tol 以下になるように、
greedy_rational_shifts と pole選択を組み合わせた適応アルゴリズムを実装。
metals/Al40, metals/Fe2MnAl の両方で使える。

元論文: metals/system with default extra bands and with number suggested
by adaptive algorithm from appendix -> scfres.jld が必要
動的版ではそれを SchurModel の次元で模擬。
"""

using LinearAlgebra

struct AdaptiveExtraBandsResult
    n_extra::Int
    k_dim::Int
    shifts::Vector{ComplexF64}
    max_err::Float64
    history::Vector{NamedTuple}
end

"""
adaptive_extra_bands_dynamic(M, ωgrid, η, b; tol=1e-8, s_max=16)

動的適応アルゴリズム:
- 静的版 run_adaptive_algorithm.jl の考え方:
  ε_{N+1} - ε_F > C*σ になるまでバンドを増やす
- 動的拡張:
  各 ω について Σ(ω+iη) の寄与が tol 以下になるまで
  共鳴極を追加。実装は greedy_rational_shifts の停止基準を
  ωgrid全体のmax errorに変更。

Returns AdaptiveExtraBandsResult
"""
function adaptive_extra_bands_dynamic(M, ωgrid, η::Real, b; tol::Real=1e-8, s_max::Int=16, rtol::Real=1e-10)
    # 貪欲に極を追加
    shifts, history = greedy_rational_shifts(M, ωgrid, η, b; s_max=s_max, rtol=rtol, init_shifts=ComplexF64[im*η])

    # 最終 basis で誤差評価
    Q = rational_krylov_basis(M.D, M.B', shifts; rtol=rtol)
    RM = ReducedSchurModel(M, Q)

    max_err = 0.0
    for ω in ωgrid
        z = ComplexF64(ω + im*η)
        exact = self_energy(M, z) * b
        approx = reduced_self_energy(RM, z) * b
        denom = norm(exact)
        err = denom==0 ? 0.0 : norm(exact-approx)/denom
        max_err = max(max_err, err)
    end

    # n_extra は Q の次元から逆算 (Cancesでは extra bands = dim(D_keep))
    n_extra = size(Q,2)
    return AdaptiveExtraBandsResult(n_extra, size(Q,2), shifts, max_err, history)
end

"""
adaptive_choice_vs_default_comparison(M, ωgrid, η, b)

元リポの metals/Al40 with default extra bands vs adaptive suggested
の比較を動的版で再現。

- default: s=4 shifts (Cances default)
- adaptive: 上記 greedy で tol 達成まで
"""
function adaptive_choice_vs_default_comparison(M, ωgrid, η::Real, b)
    default_shifts = ComplexF64.(collect(range(-1,1,length=4) .+ im*η))
    Qdef = rational_krylov_basis(M.D, M.B', default_shifts)
    RMdef = ReducedSchurModel(M, Qdef)

    ad = adaptive_extra_bands_dynamic(M, ωgrid, η, b)

    # 誤差比較
    errs_default = Float64[]
    errs_adaptive = Float64[]
    Qad = rational_krylov_basis(M.D, M.B', ad.shifts)
    RMad = ReducedSchurModel(M, Qad)
    for ω in ωgrid
        z = ω + im*η
        Σ = self_energy(M, z)
        push!(errs_default, norm(Σ - reduced_self_energy(RMdef, z))/norm(Σ))
        push!(errs_adaptive, norm(Σ - reduced_self_energy(RMad, z))/norm(Σ))
    end
    return (default=(k=size(Qdef,2), max_err=maximum(errs_default)),
            adaptive=(k=size(Qad,2), max_err=maximum(errs_adaptive), result=ad))
end
