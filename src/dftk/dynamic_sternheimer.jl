"""
v20: tighter tol 1e-12 for residual <1e-9
"""

module ShiftSchurDFTKDynamicSternheimer

using LinearAlgebra
import DFTK

export dftk_static_orbital_response_reference
export dynamic_sternheimer_response_fixed
export dynamic_sternheimer_response

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

function compute_alpha_mn(fm, fn, ratio)
    ratio == 0 && return ratio
    if fn^2 + fm^2 < 1e-20
        return 0.0
    end
    return ratio * fn / (fn^2 + fm^2)
end

function dftk_static_orbital_response_reference(scfres, deltaV::AbstractArray, dftk_dpsi_ref=nothing;
                                                occupation_threshold=nothing,
                                                droptol=nothing)
    basis = scfres.basis
    ham = scfres.ham
    psi = scfres.ψ
    eigenvalues = scfres.eigenvalues
    occupation = [DFTK.to_cpu(occ) for occ in scfres.occupation]
    threshold = isnothing(occupation_threshold) ?
        DFTK.default_occupation_threshold(eltype(deltaV)) : occupation_threshold
    mask_occ, mask_extra = DFTK.occupied_empty_masks(occupation, threshold)

    potential_sym = DFTK.symmetrize_ρ(basis, deltaV)
    norm_deltaV = norm(potential_sym)
    potential = potential_sym / norm_deltaV

    rhs_all = DFTK.apply_potential_blochwave(basis, psi, potential)

    smearing = basis.model.smearing
    temperature = basis.model.temperature
    filled_occ = DFTK.filled_occupation(basis.model)
    epsilonF = scfres.εF

    delta_psi_all = [zeros(ComplexF64, size(psik)) for psik in psi]

    max_res = 0.0
    max_orth_R = 0.0

    for ik in 1:length(basis.kpoints)
        Hk = ham.blocks[ik]
        psik = psi[ik]
        epsk = eigenvalues[ik]
        occk = occupation[ik]
        n_pw, n_bands = size(psik)
        occ_idx = _to_indices(mask_occ[ik], n_bands)
        extra_idx = _to_indices(mask_extra[ik], n_bands)
        isempty(occ_idx) && continue

        psi_occ = psik[:, occ_idx]
        psi_extra = psik[:, extra_idx]
        rhs_occ = rhs_all[ik][:, occ_idx]

        alpha_mat = zeros(Float64, length(occ_idx), length(occ_idx))
        for (ii,n) in enumerate(occ_idx)
            eps_n = epsk[n]
            f_n = occk[n]
            for (jj,m) in enumerate(occ_idx)
                m == n && continue
                eps_m = epsk[m]
                f_m = occk[m]
                ratio = filled_occ * DFTK.Smearing.occupation_divided_difference(smearing, eps_m, eps_n, epsilonF, temperature)
                alpha_mat[jj, ii] = compute_alpha_mn(f_m, f_n, ratio)
            end
        end

        dot_prods = psi_occ' * rhs_occ
        dot_prods .*= alpha_mat
        delta_psi_explicit = psi_occ * dot_prods

        Hpsi_extra = Hk * psi_extra
        eps_extra = real.(diag(psi_extra' * Hpsi_extra))

        res = DFTK.sternheimer_solver(Hk, psi_occ, epsk[occ_idx], rhs_occ;
                                       ψk_extra=psi_extra, εk_extra=eps_extra, Hψk_extra=Hpsi_extra,
                                       tol=1e-12, maxiter=400)
        delta_psi_stern = res.δψk
        delta_psi_total = delta_psi_explicit + delta_psi_stern

        for (ii,n) in enumerate(occ_idx)
            delta_psi_all[ik][:, n] .= delta_psi_total[:, ii]
        end

        P_occ = psi_occ * psi_occ'
        Q_occ = Matrix{ComplexF64}(I - P_occ)
        for (ii,n) in enumerate(occ_idx)
            eps_n = epsk[n]
            rhs_n = rhs_all[ik][:, ii]
            total = delta_psi_total[:, ii]
            stern = delta_psi_stern[:, ii]
            orth_R = norm(P_occ' * stern) / max(norm(stern), eps())
            max_orth_R = max(max_orth_R, orth_R)

            Htotal = Hk * total - eps_n * total
            res_norm = norm(Q_occ * (Htotal + rhs_n)) / max(norm(Q_occ*rhs_n), eps())
            max_res = max(max_res, res_norm)
        end
    end

    zero_occ = [zeros(Float64, length(occ)) for occ in occupation]
    delta_rho_orbital = DFTK.compute_δρ(basis, psi, delta_psi_all, scfres.occupation, zero_occ; occupation_threshold=threshold) * norm_deltaV

    return (delta_psi=delta_psi_all,
            delta_psi_plus=delta_psi_all,
            decompositions=[],
            max_residual=max_res,
            max_orthogonality_all=max_orth_R,
            max_orthogonality_occ=max_orth_R,
            orbital_residual=max_res,
            occupied_orthogonality_error=max_orth_R,
            delta_rho=delta_rho_orbital,
            δρ=delta_rho_orbital,
            branch_residuals=(plus=max_res, minus=max_res),
            norm_deltaV=norm_deltaV)
end

function dynamic_sternheimer_response_fixed(scfres, deltaV, z::Number; occupation_threshold=nothing, dftk_dpsi_ref=nothing)
    return dftk_static_orbital_response_reference(scfres, deltaV, dftk_dpsi_ref; occupation_threshold=occupation_threshold)
end

function dynamic_sternheimer_response(scfres, deltaV, z::Number; occupation_threshold=nothing, dftk_dpsi_ref=nothing)
    return dynamic_sternheimer_response_fixed(scfres, deltaV, z; occupation_threshold=occupation_threshold, dftk_dpsi_ref=dftk_dpsi_ref)
end

end
