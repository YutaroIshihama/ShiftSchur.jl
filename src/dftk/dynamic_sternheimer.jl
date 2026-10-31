"""
dynamic_sternheimer.jl — DFTK v0.8.1 chi0.jl 完全準拠の静的参照 + 動的修正版
(ShiftSchur.jl 本体用: module名を既存の ShiftSchurDFTKDynamicSternheimer に合わせる)

修正点:
- 旧実装 P = ψocc*ψocc' は誤り。DFTKは P_all = ψk*ψk' (occ+extra全部)で解く
- δψ = occ α項 + extra-band項 + Sternheimer補空間 の3成分に分解
"""

module ShiftSchurDFTKDynamicSternheimer

using LinearAlgebra
import DFTK

export dftk_static_orbital_response_reference
export dynamic_sternheimer_response_fixed
export dynamic_sternheimer_response

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

# --- Robust helper for DFTK mask handling (Bool mask vs index list vs 0/1 Int mask) ---
_to_indices(mask, n_bands::Int) = _to_indices_impl(mask, n_bands)
function _to_indices_impl(mask, n_bands::Int)
    if mask isa AbstractVector{Bool}
        return findall(mask)
    elseif eltype(mask) <: Bool
        return findall(mask)
    elseif eltype(mask) <: Integer
        # 0/1 mask of length n_bands -> convert
        if length(mask) == n_bands && all(x -> x==0 || x==1, mask)
            return findall(x -> x!=0, mask)
        else
            # already list of indices (DFTK sometimes returns Vector{Int} indices)
            return collect(Int, mask)
        end
    else
        # fallback: try findall, if fails assume Bool-like
        try
            return findall(mask)
        catch
            return findall(x -> x!=0 && x!=false, mask)
        end
    end
end

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

    potential = DFTK.symmetrize_ρ(basis, δV)
    rhs_all = DFTK.apply_potential_blochwave(basis, ψ, potential)

    δψ_ref_all = [zeros(ComplexF64, size(ψk)) for ψk in ψ]
    decompositions = OrbitalDecomposition[]

    T = eltype(basis)
    max_res = zero(real(T))
    max_orth_all = zero(real(T))
    max_orth_occ = zero(real(T))

    for (ik, kpt) in enumerate(basis.kpoints)
        ψk = ψ[ik]
        εk = ε[ik]
        n_pw, n_bands = size(ψk)
        occ_idx = _to_indices(mask_occ[ik], n_bands)
        extra_idx = _to_indices(mask_empty[ik], n_bands)
        n_occ = length(occ_idx)
        length(occ_idx)==0 && continue

        Id_pw = Matrix{ComplexF64}(I, n_pw, n_pw)
        Hk = Matrix(ham.blocks[ik] * Id_pw)

        P_all = ψk * ψk'
        Q_all = Id_pw - P_all
        ψocc = ψk[:, occ_idx]

        for n in occ_idx
            εn = εk[n]
            rhs_n = rhs_all[ik][:, n]

            δψ_occ = zeros(ComplexF64, n_pw)
            for m in occ_idx
                m == n && continue
                denom = εn - εk[m]
                abs(denom) < 1e-14 && continue
                if !isnothing(droptol)
                    f_n = occupations[ik][n]
                    f_m = occupations[ik][m]
                    abs((f_n - f_m)/denom) < droptol && continue
                end
                coeff = dot(ψk[:, m], rhs_n) / denom
                δψ_occ .+= coeff .* ψk[:, m]
            end

            δψ_extra = zeros(ComplexF64, n_pw)
            for b in extra_idx
                denom = εn - εk[b]
                abs(denom) < 1e-14 && continue
                coeff = dot(ψk[:, b], rhs_n) / denom
                δψ_extra .+= coeff .* ψk[:, b]
            end

            rhs_Q = Q_all * rhs_n
            A_proj = Q_all * (Hk - εn * Id_pw) * Q_all + P_all
            δψ_stern = A_proj \ (-rhs_Q)

            δψ_total = δψ_occ .+ δψ_extra .+ δψ_stern

            res_full = norm(Q_all * ((Hk - εn*Id_pw)*δψ_stern + rhs_n)) / max(norm(rhs_Q), eps())
            orth_all = norm(P_all' * δψ_stern) / max(norm(δψ_stern), eps())
            orth_occ = norm(ψocc' * δψ_stern) / max(norm(δψ_stern), eps())

            max_res = max(max_res, res_full)
            max_orth_all = max(max_orth_all, orth_all)
            max_orth_occ = max(max_orth_occ, orth_occ)

            δψ_ref_all[ik][:, n] .= δψ_total

            push!(decompositions, OrbitalDecomposition(
                ik, n,
                norm(δψ_occ), norm(δψ_extra), norm(δψ_stern),
                norm(δψ_total),
                0.0, 0.0,
                res_full,
                orth_occ, orth_all,
                n_occ, length(extra_idx), n_bands
            ))
        end
    end

    return (delta_psi=δψ_ref_all,
            decompositions=decompositions,
            max_residual=max_res,
            max_orthogonality_all=max_orth_all,
            max_orthogonality_occ=max_orth_occ)
end

function dynamic_sternheimer_response_fixed(scfres, δV::AbstractArray, z::Number;
                                            occupation_threshold=nothing)

    imag(z) < 0 && throw(ArgumentError("Im(z) must be >=0"))

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
    static_diags = OrbitalDecomposition[]

    for (ik, kpt) in enumerate(basis.kpoints)
        ψk = ψ[ik]
        εk = ε[ik]
        n_pw, n_bands = size(ψk)
        occ_idx = _to_indices(mask_occ[ik], n_bands)
        extra_idx = _to_indices(mask_empty[ik], n_bands)
        length(occ_idx)==0 && continue

        Id_pw = Matrix{ComplexF64}(I, n_pw, n_pw)
        Hk = Matrix(ham.blocks[ik] * Id_pw)
        P_all = ψk * ψk'
        Q_all = Id_pw - P_all

        for n in occ_idx
            εn = εk[n]
            rhs_n = rhs_all[ik][:, n]

            δψ_occ_p = zeros(ComplexF64, n_pw)
            δψ_extra_p = zeros(ComplexF64, n_pw)
            for m in occ_idx
                m==n && continue
                denom = εn - εk[m] + z
                abs(denom) < 1e-14 && continue
                δψ_occ_p .+= (dot(ψk[:, m], rhs_n)/denom) .* ψk[:, m]
            end
            for b in extra_idx
                denom = εn - εk[b] + z
                abs(denom) < 1e-14 && continue
                δψ_extra_p .+= (dot(ψk[:, b], rhs_n)/denom) .* ψk[:, b]
            end
            A_plus = Q_all * (Hk - (εn + z)*Id_pw) * Q_all + P_all
            δψ_stern_p = A_plus \ (-(Q_all*rhs_n))
            δψ_plus = δψ_occ_p .+ δψ_extra_p .+ δψ_stern_p

            z_minus = -conj(z)
            δψ_occ_m = zeros(ComplexF64, n_pw)
            δψ_extra_m = zeros(ComplexF64, n_pw)
            for m in occ_idx
                m==n && continue
                denom = εn - εk[m] + z_minus
                abs(denom) < 1e-14 && continue
                δψ_occ_m .+= (dot(ψk[:, m], rhs_n)/denom) .* ψk[:, m]
            end
            for b in extra_idx
                denom = εn - εk[b] + z_minus
                abs(denom) < 1e-14 && continue
                δψ_extra_m .+= (dot(ψk[:, b], rhs_n)/denom) .* ψk[:, b]
            end
            A_minus = Q_all * (Hk - (εn + z_minus)*Id_pw) * Q_all + P_all
            δψ_stern_m = A_minus \ (-(Q_all*rhs_n))
            δψ_minus = δψ_occ_m .+ δψ_extra_m .+ δψ_stern_m

            res_p = norm(Q_all * ((Hk - (εn+z)*Id_pw)*δψ_stern_p + rhs_n)) / max(norm(Q_all*rhs_n), eps())
            res_m = norm(Q_all * ((Hk - (εn+z_minus)*Id_pw)*δψ_stern_m + rhs_n)) / max(norm(Q_all*rhs_n), eps())
            max_res_plus = max(max_res_plus, res_p)
            max_res_minus = max(max_res_minus, res_m)
            max_orth = max(max_orth, norm(P_all'*δψ_stern_p), norm(P_all'*δψ_stern_m))

            δψ_plus_all[ik][:, n] .= δψ_plus
            δψ_minus_all[ik][:, n] .= δψ_minus

            ψr = DFTK.ifft(basis, kpt, ψk[:, n]; normalize=false)
            δψpr = DFTK.ifft(basis, kpt, δψ_plus; normalize=false)
            δψmr = DFTK.ifft(basis, kpt, δψ_minus; normalize=false)
            occ_weight = occupations[ik][n] * basis.kweights[ik]
            δρ[:, :, :, kpt.spin] .+= occ_weight .* (conj.(ψr) .* δψpr .+ ψr .* conj.(δψmr))

            if iszero(z)
                push!(static_diags, OrbitalDecomposition(
                    ik, n,
                    norm(δψ_occ_p), norm(δψ_extra_p), norm(δψ_stern_p),
                    norm(δψ_plus), 0.0, 0.0, res_p, 0.0, 0.0,
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
            static_decompositions=static_diags,
            # 旧oracle互換エイリアス
            delta_psi=δψ_plus_all,
            orbital_residual=max_res_plus,
            branch_residuals=(plus=max_res_plus, minus=max_res_minus))
end

# 旧API: そのまま fixed を呼ぶ
function dynamic_sternheimer_response(scfres, δV::AbstractArray, z::Number;
                                      occupation_threshold=nothing)
    # z=0 の時は静的参照をそのまま返す互換動作も可能だが、ここでは動的版を呼ぶ
    if iszero(z) || iszero(imag(z))
        # 静的: z=0として動的版を呼ぶ (occ+extra+stern)
        return dynamic_sternheimer_response_fixed(scfres, δV, ComplexF64(0.0);
                                                  occupation_threshold=occupation_threshold)
    else
        return dynamic_sternheimer_response_fixed(scfres, δV, z;
                                                  occupation_threshold=occupation_threshold)
    end
end

end # module
