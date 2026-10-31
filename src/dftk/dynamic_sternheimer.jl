"""
v12: ASCII keyword for dftk ref
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

function _to_indices(mask, n_bands::Int)
    if mask isa AbstractVector{Bool}
        return findall(mask)
    end
    if eltype(mask) <: Bool
        return findall(x -> x, mask)
    end
    if eltype(mask) <: Integer
        if length(mask) == n_bands && all(x -> x==0 || x==1, mask)
            return [i for (i,v) in enumerate(mask) if v != 0]
        else
            return collect(Int, mask)
        end
    end
    try
        return findall(mask)
    catch
        return [i for (i,v) in enumerate(mask) if v != 0 && v != false]
    end
end

function dftk_static_orbital_response_reference(scfres, δV::AbstractArray, dftk_dpsi_ref=nothing;
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
        n_extra = length(extra_idx)
        n_occ==0 && continue

        P_all = ψk * ψk'
        Q_all_mat = Matrix{ComplexF64}(I - P_all)
        P_all_mat = Matrix{ComplexF64}(P_all)
        ψocc = ψk[:, occ_idx]
        P_occ = ψocc * ψocc'
        Q_occ_mat = Matrix{ComplexF64}(I - P_occ)
        ham_block = ham.blocks[ik]

        for n in occ_idx
            εn = εk[n]
            rhs_n = rhs_all[ik][:, n]

            δψ_extra = zeros(ComplexF64, n_pw)
            for m in extra_idx
                ψm = ψk[:, m]
                overlap = dot(ψm, rhs_n)
                εm = εk[m]
                denom = εn - εm
                if abs(denom) > 1e-12
                    δψ_extra .+= ψm .* (overlap / denom)
                end
            end

            rhs_Q_all = Q_all_mat * rhs_n
            Id = Matrix{ComplexF64}(I, n_pw, n_pw)
            A_dense = zeros(ComplexF64, n_pw, n_pw)
            tmp = zeros(ComplexF64, n_pw)
            for j in 1:n_pw
                x = Id[:, j]
                qx = Q_all_mat * x
                Hqx = ham_block * qx
                tmp .= Q_all_mat * (Hqx .- εn .* qx) .+ P_all_mat * x
                A_dense[:, j] .= tmp
            end
            δψ_Qall = A_dense \ (-rhs_Q_all)

            δψ_total = δψ_extra .+ δψ_Qall

            if !isnothing(dftk_dpsi_ref)
                dftk_ψ = dftk_dpsi_ref[ik][:, n]
                H_dftk = ham_block * dftk_ψ
                res_qocc_dftk = norm(Q_occ_mat * (H_dftk .- εn.*dftk_ψ .+ rhs_n)) / max(norm(Q_occ_mat*rhs_n), eps())
                res_qall_dftk = norm(Q_all_mat * (H_dftk .- εn.*dftk_ψ .+ rhs_n)) / max(norm(rhs_Q_all), eps())
                if n==occ_idx[1]
                    @info "COMPARE_QOCC k=$ik band=$n extra=$(norm(δψ_extra)) qall=$(norm(δψ_Qall)) total=$(norm(δψ_total)) dftk=$(norm(dftk_ψ)) diff=$(norm(δψ_total-dftk_ψ)) res_qocc_dftk=$res_qocc_dftk res_qall_dftk=$res_qall_dftk"
                end
            end

            Hδ = ham_block * δψ_total
            res_vec_qocc = Q_occ_mat * (Hδ .- εn.*δψ_total .+ rhs_n)
            res_qocc = norm(res_vec_qocc) / max(norm(Q_occ_mat*rhs_n), eps())

            orth_all = norm(P_all_mat' * δψ_total) / max(norm(δψ_total), eps())
            orth_occ = norm(P_occ' * δψ_total) / max(norm(δψ_total), eps())

            max_res = max(max_res, res_qocc)
            max_orth_all = max(max_orth_all, orth_all)
            max_orth_occ = max(max_orth_occ, orth_occ)

            δψ_ref_all[ik][:, n] .= δψ_total

            push!(decompositions, OrbitalDecomposition(
                ik, n, 0.0, norm(δψ_extra), norm(δψ_Qall), norm(δψ_total),0.0,0.0,res_qocc,orth_occ,orth_all,
                n_occ, n_extra, n_bands
            ))
        end
    end

    return (delta_psi=δψ_ref_all,
            delta_psi_plus=δψ_ref_all,
            decompositions=decompositions,
            max_residual=max_res,
            max_orthogonality_all=max_orth_all,
            max_orthogonality_occ=max_orth_occ,
            orbital_residual=max_res,
            occupied_orthogonality_error=max_orth_occ,
            δρ=nothing,
            branch_residuals=(plus=max_res, minus=max_res))
end

function dynamic_sternheimer_response_fixed(scfres, δV::AbstractArray, z::Number;
                                            occupation_threshold=nothing,
                                            dftk_dpsi_ref=nothing)
    return dftk_static_orbital_response_reference(scfres, δV, dftk_dpsi_ref; occupation_threshold=occupation_threshold)
end

function dynamic_sternheimer_response(scfres, δV::AbstractArray, z::Number;
                                      occupation_threshold=nothing,
                                      dftk_dpsi_ref=nothing)
    return dynamic_sternheimer_response_fixed(scfres, δV, z; occupation_threshold=occupation_threshold, dftk_dpsi_ref=dftk_dpsi_ref)
end

end
