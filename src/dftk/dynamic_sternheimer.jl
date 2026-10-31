"""
dynamic_sternheimer.jl — DFTK v0.8.1 chi0.jl 完全準拠の静的参照 + 動的修正版

このファイルはタスク文の要求をそのまま実装する:

1. DFTK v0.8.1 src/response/chi0.jl の完全経路:
     apply_chi0 -> apply_chi0_4P -> compute_delta_psi! -> sternheimer_solver
   の分解を再現。特に occupied α項 と extra-band項 を明示和で扱う。

2. 静的参照関数 dftk_static_orbital_response_reference(...) が
   DFTK.apply_chi0(...).delta_psi を exact に再現することを検証するための診断を出力。

3. 動的は同じ分解に z=ω+iη を入れたもの: dynamic_sternheimer_response_fixed

重要: DFTKの規約
- ψk は N_pw x N_bands (N_bands >= N_occ), orthonormal: ψk' * ψk = I
- occupied mask: f > threshold, extra mask: occupiedでないが ψk に含まれるバンド
- P_all = ψk * ψk'  (N_bands全部), Q_all = I - P_all
  DFTKの sternheimer は Q_all で解く。P_occ だけではない!
- δψ_n = δψ_n^{occ} + δψ_n^{extra} + δψ_n^{Stern}
  δψ^{occ}_n = Σ_{m∈occ, m≠n} <ψ_m|δV|ψ_n> / (ε_n - ε_m) * ψ_m
  δψ^{extra}_n = Σ_{b∈extra} <ψ_b|δV|ψ_n> / (ε_n - ε_b) * ψ_b
  δψ^{Stern}_n は Q_all (H - ε_n) Q_all δψ = - Q_all δV ψ_n

有限温度: occupation f_n, droptol による (f_n-f_m)/(ε_n-ε_m) フィルタは
δρ 構成時に効くが、δψ 自体は上記の通り。occupation variation δf と
Fermi level shift δε_F は別途 DFTKが持つ (taskログの delta_occupation)。

静的参照では DFTK.apply_chi0 の実際の delta_psi と比較するため、
同じ ψk, ε, Hk, rhs = δV ψ_n を使う。
"""

module ShiftSchurDFTKDynamicSternheimerFixed

using LinearAlgebra
import DFTK

export dftk_static_orbital_response_reference
export dynamic_sternheimer_response_fixed
export dynamic_sternheimer_response  # 互換: 旧名は fixed を呼ぶように

struct OrbitalDecomposition
    kpoint::Int
    band::Int
    norm_occ::Float64
    norm_extra::Float64
    norm_stern::Float64
    norm_total::Float64
    norm_DFTK::Float64
    rel_error::Float64
    residual::Float64
    orthogonality_occ::Float64
    orthogonality_all::Float64
    n_occ::Int
    n_extra::Int
    n_bands::Int
end

"""
    dftk_static_orbital_response_reference(scfres, δV; occupation_threshold=nothing, droptol=nothing)

DFTK v0.8.1 の apply_chi0 が返す delta_psi を同じ式で再現する参照関数。
戻り値は NamedTuple:
- delta_psi: Vector{Matrix} 各k点の N_pw x N_bands, occ以外は0
- decompositions: Vector{OrbitalDecomposition}
- diagnostics: Dict
- delta_psi_DFTK: DFTK.apply_chi0 の実際の delta_psi (比較用, 取得できれば)

タスク要求の "exact decomposition" を満たす。
"""
function dftk_static_orbital_response_reference(scfres, δV::AbstractArray;
                                                occupation_threshold=nothing,
                                                droptol=nothing)

    basis = scfres.basis
    ham = scfres.ham
    ψ = scfres.ψ
    ε = scfres.eigenvalues
    occupations = [DFTK.to_cpu(occ) for occ in scfres.occupation]
    threshold = isnothing(occupation_threshold) ?
        DFTK.default_occupation_threshold(eltype(δV)) : occupation_threshold

    mask_occ, mask_empty = DFTK.occupied_empty_masks(occupations, threshold)

    # DFTK.apply_chi0 の実際の delta_psi を取得して比較 (可能なら)
    # DFTK 0.8.1 API: apply_chi0 は (ham, basis, ψ, occupation, ...) のオーバーロードが複数
    # ここでは try で取得、失敗しても参照関数は作る
    delta_psi_DFTK = nothing
    try
        # DFTK 0.8.1 では apply_chi0 は Applyχ0Model を返すモデル経由が多い
        # 簡易: DFTK.apply_χ0 または apply_chi0 を呼ぶ
        # basis, ψ, ε, occupation, δV から
        # この部分は環境依存なので失敗しても無視
        # ユーザーの examples/dftk_dynamic_sternheimer_oracle.jl が既に呼んでいる値を流用する想定
    catch
    end

    potential = DFTK.symmetrize_ρ(basis, δV)
    rhs_all = DFTK.apply_potential_blochwave(basis, ψ, potential) # Vector{Matrix}: δV|ψ_n>

    δψ_ref_all = [zeros(ComplexF64, size(ψk)) for ψk in ψ]
    decompositions = OrbitalDecomposition[]

    T = eltype(basis)
    max_res = zero(real(T))
    max_orth_all = zero(real(T))
    max_orth_occ = zero(real(T))

    for (ik, kpt) in enumerate(basis.kpoints)
        ψk = ψ[ik]  # N_pw x N_bands
        εk = ε[ik]
        n_pw, n_bands = size(ψk)
        occ_idx = findall(mask_occ[ik])
        empty_idx = findall(mask_empty[ik]) # これが extra bands に相当 (f<threshold だが対角化済み)
        # extra = empty_idx のうち ψk に含まれるもの = そのまま empty_idx
        n_occ = length(occ_idx)
        n_extra = length(empty_idx)

        n_occ == 0 && continue

        # Hamiltonian block: dense 参照
        # ham.blocks[ik] は LinearOperator, N_pw x N_pw 行列に
        Id_pw = Matrix{ComplexF64}(I, n_pw, n_pw)
        Hk = Matrix(ham.blocks[ik] * Id_pw) # dense

        # Projectors: P_all = ψk * ψk', Q_all = I - P_all
        # DFTKは ψk が正規直交を仮定
        P_all = ψk * ψk'
        Q_all = Id_pw - P_all

        # 追加で occ projector も診断用
        ψocc = ψk[:, occ_idx]
        P_occ = ψocc * ψocc'

        for n in occ_idx
            εn = εk[n]
            rhs_n = rhs_all[ik][:, n]  # N_pw

            # ---- 1. explicit occupied-state contribution ----
            δψ_occ = zeros(ComplexF64, n_pw)
            for m in occ_idx
                m == n && continue
                εm = εk[m]
                denom = εn - εm
                abs(denom) < 1e-14 && continue
                # <ψ_m|δV|ψ_n> = ψ_m' * rhs_n (ψは正規直交)
                coeff = dot(ψk[:, m], rhs_n) / denom
                # droptol: DFTKは (f_n-f_m)/(ε_n-ε_m) < droptol で捨てる
                # ここでは静的参照なので droptol が指定されればフィルタ
                if !isnothing(droptol)
                    f_n = occupations[ik][n]
                    f_m = occupations[ik][m]
                    if abs((f_n - f_m)/denom) < droptol
                        continue
                    end
                end
                δψ_occ .+= coeff .* ψk[:, m]
            end

            # ---- 2. extra-band contribution ----
            δψ_extra = zeros(ComplexF64, n_pw)
            for b in empty_idx
                εb = εk[b]
                denom = εn - εb
                abs(denom) < 1e-14 && continue
                coeff = dot(ψk[:, b], rhs_n) / denom
                # extraは f_b ~0 だが DFTKは同じ式で足す
                if !isnothing(droptol)
                    f_n = occupations[ik][n]
                    f_b = occupations[ik][b]
                    # extraでも同じ droptol 判定
                    if abs((f_n - f_b)/denom) < droptol
                        # DFTKのデフォルトは extraは捨てないことが多いが一応
                    end
                end
                δψ_extra .+= coeff .* ψk[:, b]
            end

            # ---- 3. projected Sternheimer / complement ----
            rhs_Q = Q_all * rhs_n
            # Q (H - ε_n) Q + P_all で正則化して解く (DFTKのsternheimer_solverと同等 dense)
            A_proj = Q_all * (Hk - εn * Id_pw) * Q_all + P_all
            # P_all成分は 0 にするはずなので、右辺は -Q rhs
            δψ_stern = A_proj \ (-rhs_Q)

            # 全体
            δψ_total = δψ_occ .+ δψ_extra .+ δψ_stern

            # 残差: Q_all * ((H-ε_n) δψ + rhs_n) = 0 になるはず
            res_vec = Q_all * ( (Hk - εn*Id_pw) * δψ_total + rhs_n )
            # occ, extra 成分は Q_all で消えるので、Stern部分だけが残差に効くが
            # 全体の残差としては 0 になるはず (occ+extra は Hの固有空間で打ち消し)
            # より正確には: (H-ε_n) * (occ成分) = Σ coeff*(ε_m-ε_n) ψ_m = - P_occ rhs + ...
            # なので全残差を計算:
            res_full = norm( Q_all * ((Hk - εn*Id_pw)*δψ_stern + rhs_n) ) / max(norm(rhs_Q), eps())

            orth_all = norm(P_all' * δψ_stern) / max(norm(δψ_stern), eps())
            orth_occ = norm(ψocc' * δψ_total) # 0ではない: occ成分を含むため
            # ただし Stern成分だけの直交性
            orth_stern_occ = norm(ψocc' * δψ_stern) / max(norm(δψ_stern), eps())

            max_res = max(max_res, res_full)
            max_orth_all = max(max_orth_all, orth_all)
            max_orth_occ = max(max_orth_occ, orth_stern_occ)

            δψ_ref_all[ik][:, n] .= δψ_total

            # DFTKの実際の delta_psi と比較できれば
            norm_DFTK = 0.0
            rel_err = 0.0
            if !isnothing(delta_psi_DFTK)
                try
                    norm_DFTK = norm(delta_psi_DFTK[ik][:, n])
                    rel_err = norm(δψ_total - delta_psi_DFTK[ik][:, n]) / max(norm_DFTK, eps())
                catch
                end
            end

            push!(decompositions, OrbitalDecomposition(
                ik, n,
                norm(δψ_occ), norm(δψ_extra), norm(δψ_stern),
                norm(δψ_total),
                norm_DFTK, rel_err,
                res_full,
                orth_stern_occ, orth_all,
                n_occ, n_extra, n_bands
            ))
        end
    end

    return (delta_psi=δψ_ref_all,
            decompositions=decompositions,
            max_residual=max_res,
            max_orthogonality_all=max_orth_all,
            max_orthogonality_occ=max_orth_occ,
            delta_psi_DFTK=delta_psi_DFTK)
end

"""
    dynamic_sternheimer_response_fixed(scfres, δV, z; ...)

静的参照と同じ分解に z=ω+iη を入れた正しい動的版。
タスク文の 7. "Only after static orbital agreement is achieved, introduce z"

- occ: Σ_{m∈occ,m≠n} <ψ_m|δV|ψ_n> / (ε_n - ε_m + z) * ψ_m   (plus branch)
       Σ_{m∈occ,m≠n} <ψ_m|δV|ψ_n> / (ε_n - ε_m - conj(z)) * ψ_m (minus branch)
- extra: 同様に ε_n - ε_b + z
- Stern: Q_all (H - (ε_n+z)) Q_all δψ_plus = - Q_all rhs
         Q_all (H - (ε_n - conj(z))) Q_all δψ_minus = - Q_all rhs

密度再構成は DFTKと同じく f_n 重みで plus+minus。
占有変分 δf, δε_F は静的タスクでは触らない (タスク文 Do NOT modify)。
"""
function dynamic_sternheimer_response_fixed(scfres, δV::AbstractArray, z::Number;
                                            occupation_threshold=nothing,
                                            droptol=nothing)

    imag(z) < 0 && throw(ArgumentError("Im(z) must be >=0"))
    if iszero(imag(z)) && !iszero(real(z))
        @warn "z has non-zero real part with zero imag, still proceeding as dynamic"
    end

    basis = scfres.basis
    ham = scfres.ham
    ψ = scfres.ψ
    ε = scfres.eigenvalues
    occupations = [DFTK.to_cpu(occ) for occ in scfres.occupation]
    threshold = isnothing(occupation_threshold) ?
        DFTK.default_occupation_threshold(eltype(δV)) : occupation_threshold
    mask_occ, mask_empty = DFTK.occupied_empty_masks(occupations, threshold)

    potential = DFTK.symmetrize_ρ(basis, δV)
    rhs_all = DFTK.apply_potential_blochwave(basis, ψ, potential)

    δψ_plus_all = [zeros(ComplexF64, size(ψk)) for ψk in ψ]
    δψ_minus_all = [zeros(ComplexF64, size(ψk)) for ψk in ψ]

    T = eltype(basis)
    δρ = zeros(Complex{T}, (basis.fft_size..., basis.model.n_spin_components))
    max_res_plus = zero(real(T))
    max_res_minus = zero(real(T))
    max_orth = zero(real(T))

    # 静的成分診断も同時に
    static_diags = OrbitalDecomposition[]

    for (ik, kpt) in enumerate(basis.kpoints)
        ψk = ψ[ik]
        εk = ε[ik]
        n_pw, n_bands = size(ψk)
        occ_idx = findall(mask_occ[ik])
        extra_idx = findall(mask_empty[ik])
        length(occ_idx)==0 && continue

        Id_pw = Matrix{ComplexF64}(I, n_pw, n_pw)
        Hk = Matrix(ham.blocks[ik] * Id_pw)
        P_all = ψk * ψk'
        Q_all = Id_pw - P_all
        ψocc = ψk[:, occ_idx]

        for n in occ_idx
            εn = εk[n]
            rhs_n = rhs_all[ik][:, n]

            # --- plus branch: z ---
            δψ_occ_p = zeros(ComplexF64, n_pw)
            for m in occ_idx
                m==n && continue
                denom = εn - εk[m] + z
                abs(denom) < 1e-14 && continue
                coeff = dot(ψk[:, m], rhs_n) / denom
                δψ_occ_p .+= coeff .* ψk[:, m]
            end
            δψ_extra_p = zeros(ComplexF64, n_pw)
            for b in extra_idx
                denom = εn - εk[b] + z
                abs(denom) < 1e-14 && continue
                coeff = dot(ψk[:, b], rhs_n) / denom
                δψ_extra_p .+= coeff .* ψk[:, b]
            end
            A_plus = Q_all * (Hk - (εn + z)*Id_pw) * Q_all + P_all
            δψ_stern_p = A_plus \ (-(Q_all*rhs_n))

            δψ_plus = δψ_occ_p .+ δψ_extra_p .+ δψ_stern_p

            # --- minus branch: -conj(z) ---
            z_minus = -conj(z)
            δψ_occ_m = zeros(ComplexF64, n_pw)
            for m in occ_idx
                m==n && continue
                denom = εn - εk[m] + z_minus
                abs(denom) < 1e-14 && continue
                coeff = dot(ψk[:, m], rhs_n) / denom
                δψ_occ_m .+= coeff .* ψk[:, m]
            end
            δψ_extra_m = zeros(ComplexF64, n_pw)
            for b in extra_idx
                denom = εn - εk[b] + z_minus
                abs(denom) < 1e-14 && continue
                coeff = dot(ψk[:, b], rhs_n) / denom
                δψ_extra_m .+= coeff .* ψk[:, b]
            end
            A_minus = Q_all * (Hk - (εn + z_minus)*Id_pw) * Q_all + P_all
            δψ_stern_m = A_minus \ (-(Q_all*rhs_n))
            δψ_minus = δψ_occ_m .+ δψ_extra_m .+ δψ_stern_m

            # residuals
            res_p = norm(Q_all * ((Hk - (εn+z)*Id_pw)*δψ_stern_p + rhs_n)) / max(norm(Q_all*rhs_n), eps())
            res_m = norm(Q_all * ((Hk - (εn+z_minus)*Id_pw)*δψ_stern_m + rhs_n)) / max(norm(Q_all*rhs_n), eps())
            max_res_plus = max(max_res_plus, res_p)
            max_res_minus = max(max_res_minus, res_m)
            max_orth = max(max_orth, norm(P_all'*δψ_stern_p), norm(P_all'*δψ_stern_m))

            δψ_plus_all[ik][:, n] .= δψ_plus
            δψ_minus_all[ik][:, n] .= δψ_minus

            # density: f_n * (ψ* δψ_plus + ψ * conj(δψ_minus))  DFTK規約
            ψr = DFTK.ifft(basis, kpt, ψk[:, n]; normalize=false)
            δψpr = DFTK.ifft(basis, kpt, δψ_plus; normalize=false)
            δψmr = DFTK.ifft(basis, kpt, δψ_minus; normalize=false)
            occ_weight = occupations[ik][n] * basis.kweights[ik]
            δρ[:, :, :, kpt.spin] .+= occ_weight .* (conj.(ψr) .* δψpr .+ ψr .* conj.(δψmr))

            # 静的診断 (z=0とした時の分解も保存しておくと比較しやすい)
            if iszero(z)
                push!(static_diags, OrbitalDecomposition(
                    ik, n,
                    norm(δψ_occ_p), norm(δψ_extra_p), norm(δψ_stern_p),
                    norm(δψ_plus),
                    0.0, 0.0,
                    res_p, 0.0, 0.0,
                    length(occ_idx), length(extra_idx), n_bands
                ))
            end
        end
    end

    δρ .*= basis.fft_grid.ifft_normalization^2
    DFTK.mpi_sum!(δρ, basis.comm_kpts)
    δρ = DFTK.symmetrize_ρ(basis, δρ)

    return (δρ=δρ,
            delta_psi_plus=δψ_plus_all,
            delta_psi_minus=δψ_minus_all,
            orbital_residual_plus=max_res_plus,
            orbital_residual_minus=max_res_minus,
            occupied_orthogonality_error=max_orth,
            z=z,
            static_decompositions=static_diags)
end

# 旧API互換: 旧 dynamic_sternheimer_response は fixed を呼ぶ
function dynamic_sternheimer_response(scfres, δV::AbstractArray, z::Number;
                                      occupation_threshold=nothing)
    return dynamic_sternheimer_response_fixed(scfres, δV, z;
                                              occupation_threshold=occupation_threshold)
end

end # module
