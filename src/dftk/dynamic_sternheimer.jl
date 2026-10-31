module ShiftSchurDFTKDynamicSternheimer

using LinearAlgebra
import DFTK

export dynamic_sternheimer_response

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
        occ_indices = findall(mask_occ[ik])
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
