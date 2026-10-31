module ShiftSchurDFTKDynamicSternheimer

using LinearAlgebra
import DFTK

export dynamic_sternheimer_response, dftk_static_orbital_response_reference

"""
    dftk_static_orbital_response_reference(scfres, δV; tol=1e-9)

Reconstruct DFTK 0.8.1's q=0 static orbital response using its explicit
occupied-state contribution and `sternheimer_solver`, including the
extra-band Schur-complement treatment. The perturbation is symmetrized and
normalized as in `DFTK.apply_χ0`.
"""
function dftk_static_orbital_response_reference(scfres, δV::AbstractArray;
                                                tol=1e-9,
                                                occupation_threshold=scfres.occupation_threshold)
    basis = scfres.basis
    ψ = scfres.ψ
    ε = scfres.eigenvalues
    occupations = [DFTK.to_cpu(occ) for occ in scfres.occupation]
    mask_occ, mask_extra = DFTK.occupied_empty_masks(
        occupations, occupation_threshold)

    potential = DFTK.symmetrize_ρ(basis, δV)
    normδH = norm(potential)
    normδH > eps(eltype(basis)) ||
        throw(ArgumentError("static orbital response requires nonzero δV"))
    potential ./= normδH
    rhs = DFTK.apply_potential_blochwave(basis, ψ, potential)

    bandtolalg = DFTK.BandtolBalanced(
        basis, scfres.ψ, scfres.occupation; occupation_threshold)
    band_tolerances = DFTK.determine_band_tolerances(bandtolalg, tol)
    filled_occ = DFTK.filled_occupation(basis.model)
    temperature = basis.model.temperature
    smearing = basis.model.smearing
    δψ_all = zero.(rhs)
    per_kpoint = NamedTuple[]

    for ik in eachindex(basis.kpoints)
        ψk = ψ[ik]
        occ_indices = collect(mask_occ[ik])
        extra_indices = collect(mask_extra[ik])
        ψocc = ψk[:, occ_indices]
        ψextra = ψk[:, extra_indices]
        εocc = ε[ik][occ_indices]
        rhs_occ = rhs[ik][:, occ_indices]
        nocc = length(occ_indices)

        if nocc == 0
            push!(per_kpoint, (kpoint=ik, n_occupied=0,
                               n_extra=length(extra_indices),
                               explicit_norm=0.0, extra_norm=0.0,
                               complement_norm=0.0, residual=0.0,
                               cg_residual=0.0,
                               reference_norm=0.0))
            continue
        end

        α = zeros(eltype(rhs_occ), nocc, nocc)
        for n in 1:nocc
            fn = filled_occ * DFTK.Smearing.occupation(
                smearing, (εocc[n] - scfres.εF) / temperature)
            for m in 1:nocc
                m == n && continue
                fm = filled_occ * DFTK.Smearing.occupation(
                    smearing, (εocc[m] - scfres.εF) / temperature)
                ratio = filled_occ * DFTK.Smearing.occupation_divided_difference(
                    smearing, εocc[m], εocc[n], scfres.εF, temperature)
                α[m, n] = DFTK.compute_αmn(fm, fn, ratio)
            end
        end
        explicit = ψocc * (α .* (ψocc' * rhs_occ))

        Hk = scfres.ham.blocks[ik]
        Hψextra = Hk * ψextra
        εextra = diag(real.(ψextra' * Hψextra))
        stern = DFTK.sternheimer_solver(
            Hk, ψocc, εocc, rhs_occ;
            ψk_extra=ψextra, εk_extra=εextra, Hψk_extra=Hψextra,
            tol=band_tolerances[ik])

        solver_response = stern.δψk
        extra = ψextra * (ψextra' * solver_response)
        complement = solver_response - extra
        total = explicit + extra + complement
        δψ_all[ik][:, occ_indices] .= total

        P = ψocc * ψocc'
        Q = I - P
        residuals = [norm(Q * ((Hk * total[:, n]) -
                                εocc[n] .* total[:, n] + rhs_occ[:, n])) /
                     max(norm(Q * rhs_occ[:, n]), eps(eltype(basis)))
                     for n in 1:nocc]
        push!(per_kpoint, (
            kpoint=ik, n_occupied=nocc, n_extra=length(extra_indices),
            explicit_norm=norm(explicit), extra_norm=norm(extra),
            complement_norm=norm(complement), residual=maximum(residuals),
            cg_residual=maximum(stern.residual_norms),
            reference_norm=norm(total)))
    end

    return (delta_psi=δψ_all, per_kpoint=per_kpoint,
            normalized_perturbation_norm=normδH)
end

"""
    dynamic_sternheimer_response(scfres, δV, z; occupation_threshold=nothing)

Dense reference implementation of the q=0 frequency-dependent independent
particle response for `z = ω + im*η`. It uses both shifted Sternheimer
branches and the occupied projector from the DFTK SCF result. This is an
oracle for small systems, not an optimized solver or a dielectric spectrum.
"""
function dynamic_sternheimer_response(scfres, δV::AbstractArray, z::Number;
                                      occupation_threshold=nothing)
    iszero(imag(z)) && !iszero(real(z)) &&
        throw(ArgumentError("dynamic Sternheimer reference requires Im(z) > 0"))
    imag(z) >= 0 || throw(ArgumentError("Im(z) must be nonnegative"))

    basis = scfres.basis
    ham = scfres.ham
    ψ = scfres.ψ
    ε = scfres.eigenvalues
    occupations = [DFTK.to_cpu(occ) for occ in scfres.occupation]
    threshold = isnothing(occupation_threshold) ?
        DFTK.default_occupation_threshold(eltype(δV)) : occupation_threshold
    mask_occ, _ = DFTK.occupied_empty_masks(occupations, threshold)

    potential = DFTK.symmetrize_ρ(basis, δV)
    rhs = DFTK.apply_potential_blochwave(basis, ψ, potential)
    δψ_plus_all = zero.(rhs)
    T = eltype(basis)
    δρ = zeros(Complex{T}, (basis.fft_size..., basis.model.n_spin_components))
    max_residual = zero(T)
    max_orthogonality = zero(T)
    max_branch_residuals = (plus=zero(T), minus=zero(T))

    for (ik, kpt) in enumerate(basis.kpoints)
        ψk = ψ[ik]
        occ_indices = collect(mask_occ[ik])
        ψocc = ψk[:, occ_indices]
        size(ψocc, 2) == 0 && continue

        n_pw = size(ψk, 1)
        identity = Matrix{eltype(ψk)}(I, n_pw, n_pw)
        Hk = ham.blocks[ik] * identity
        P = ψocc * ψocc'
        Q = identity - P

        for n in occ_indices
            rhs_n = rhs[ik][:, n]
            rhs_projected = Q * rhs_n
            εn = ε[ik][n]
            branch_results = map((z, -conj(z))) do shift
                shifted_H = Hk - (εn + shift) * identity
                projected_H = Q * shifted_H * Q + P
                δψn = projected_H \ (-rhs_projected)
                residual = norm(Q * (shifted_H * δψn + rhs_n)) /
                           max(norm(rhs_projected), eps(T))
                orthogonality = norm(ψocc' * δψn) /
                                max(norm(δψn), eps(T))
                (δψ=δψn, residual=residual, orthogonality=orthogonality)
            end
            plus, minus = branch_results
            δψ_plus_all[ik][:, n] .= plus.δψ
            max_residual = max(max_residual, plus.residual, minus.residual)
            max_branch_residuals = (
                plus=max(max_branch_residuals.plus, plus.residual),
                minus=max(max_branch_residuals.minus, minus.residual))
            max_orthogonality = max(max_orthogonality,
                                    plus.orthogonality, minus.orthogonality)

            ψr = DFTK.ifft(basis, kpt, ψk[:, n]; normalize=false)
            δψ_plus_r = DFTK.ifft(basis, kpt, plus.δψ; normalize=false)
            δψ_minus_r = DFTK.ifft(basis, kpt, minus.δψ; normalize=false)
            occ_weight = occupations[ik][n] * basis.kweights[ik]
            δρ[:, :, :, kpt.spin] .+= occ_weight .* (
                conj.(ψr) .* δψ_plus_r .+ ψr .* conj.(δψ_minus_r))
        end
    end

    δρ .*= basis.fft_grid.ifft_normalization^2
    DFTK.mpi_sum!(δρ, basis.comm_kpts)
    δρ = DFTK.symmetrize_ρ(basis, δρ)

    return (δρ=δρ, delta_psi_plus=δψ_plus_all,
            orbital_residual=max_residual,
            branch_residuals=max_branch_residuals,
            occupied_orthogonality_error=max_orthogonality, z=z)
end

end # module
