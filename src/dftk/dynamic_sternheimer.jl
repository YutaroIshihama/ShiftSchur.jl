"""
v8: v7のDimensionMismatch修正
- @. tmp = Q*... は * が .* にブロードキャストされて落ちるので .= に修正
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
        n_occ==0 && continue

        P_all = ψk * ψk'
        Q_all = I - P_all
        Q_mat = Matrix{ComplexF64}(Q_all)
        P_mat = Matrix{ComplexF64}(P_all)
        ψocc = ψk[:, occ_idx]
        P_occ = ψocc * ψocc'

        ham_block = ham.blocks[ik]

        for n in occ_idx
            εn = εk[n]
            rhs_n = rhs_all[ik][:, n]
            rhs_Q = Q_mat * rhs_n

            δψ_occ = zeros(ComplexF64, n_pw)
            δψ_extra = zeros(ComplexF64, n_pw)
            for m in occ_idx
                m==n && continue
                denom = εn - εk[m]
                abs(denom)<1e-14 && continue
                if !isnothing(droptol)
                    f_n = occupations[ik][n]
                    f_m = occupations[ik][m]
                    abs((f_n - f_m)/denom) < droptol && continue
                end
                δψ_occ .+= dot(ψk[:,m], rhs_n)/denom .* ψk[:,m]
            end
            for b in extra_idx
                denom = εn - εk[b]
                abs(denom)<1e-14 && continue
                δψ_extra .+= dot(ψk[:,b], rhs_n)/denom .* ψk[:,b]
            end

            Id = Matrix{ComplexF64}(I, n_pw, n_pw)
            A_dense = zeros(ComplexF64, n_pw, n_pw)
            tmp = zeros(ComplexF64, n_pw)
            for j in 1:n_pw
                x = Id[:, j]
                qx = Q_mat * x
                Hqx = ham_block * qx
                tmp .= Q_mat * (Hqx .- εn .* qx) .+ P_mat * x
                A_dense[:, j] .= tmp
            end
            δψ_stern = A_dense \ (-rhs_Q)

            δψ_total = δψ_stern

            Hδ = ham_block * δψ_stern
            res_vec = Q_mat * (Hδ .- εn.*δψ_stern .+ rhs_n)
            res = norm(res_vec) / max(norm(rhs_Q), eps())
            orth_all = norm(P_mat' * δψ_stern) / max(norm(δψ_stern), eps())
            orth_occ = norm(P_occ' * δψ_total) / max(norm(δψ_total), eps())

            max_res = max(max_res, res)
            max_orth_all = max(max_orth_all, orth_all)
            max_orth_occ = max(max_orth_occ, orth_occ)

            δψ_ref_all[ik][:, n] .= δψ_total

            push!(decompositions, OrbitalDecomposition(
                ik, n,
                norm(δψ_occ), norm(δψ_extra), norm(δψ_stern),
                norm(δψ_total),0.0,0.0,res,orth_occ,orth_all,
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

        P_all = ψk * ψk'
        Q_all = I - P_all
        Q_mat = Matrix{ComplexF64}(Q_all)
        P_mat = Matrix{ComplexF64}(P_all)
        P_occ = ψk[:, occ_idx] * ψk[:, occ_idx]'
        ham_block = ham.blocks[ik]

        for n in occ_idx
            εn = εk[n]
            rhs_n = rhs_all[ik][:, n]
            rhs_Q = Q_mat * rhs_n

            Id = Matrix{ComplexF64}(I, n_pw, n_pw)

            A_plus = zeros(ComplexF64, n_pw, n_pw)
            tmp = zeros(ComplexF64, n_pw)
            for j in 1:n_pw
                x = Id[:, j]
                qx = Q_mat * x
                Hqx = ham_block * qx
                tmp .= Q_mat * (Hqx .- (εn+z).*qx) .+ P_mat * x
                A_plus[:, j] .= tmp
            end
            δψ_stern_p = A_plus \ (-rhs_Q)
            δψ_plus = δψ_stern_p

            z_minus = -conj(z)
            A_minus = zeros(ComplexF64, n_pw, n_pw)
            for j in 1:n_pw
                x = Id[:, j]
                qx = Q_mat * x
                Hqx = ham_block * qx
                tmp .= Q_mat * (Hqx .- (εn+z_minus).*qx) .+ P_mat * x
                A_minus[:, j] .= tmp
            end
            δψ_stern_m = A_minus \ (-rhs_Q)
            δψ_minus = δψ_stern_m

            Hδp = ham_block * δψ_stern_p
            res_p = norm(Q_mat * (Hδp .- (εn+z).*δψ_stern_p .+ rhs_n)) / max(norm(rhs_Q), eps())
            Hδm = ham_block * δψ_stern_m
            res_m = norm(Q_mat * (Hδm .- (εn+z_minus).*δψ_stern_m .+ rhs_n)) / max(norm(rhs_Q), eps())

            max_res_plus = max(max_res_plus, res_p)
            max_res_minus = max(max_res_minus, res_m)
            max_orth = max(max_orth, norm(P_occ'*δψ_plus), norm(P_occ'*δψ_minus))

            δψ_plus_all[ik][:, n] .= δψ_plus
            δψ_minus_all[ik][:, n] .= δψ_minus

            ψr = DFTK.ifft(basis, kpt, ψk[:, n]; normalize=false)
            δψpr = DFTK.ifft(basis, kpt, δψ_plus; normalize=false)
            δψmr = DFTK.ifft(basis, kpt, δψ_minus; normalize=false)
            occ_weight = occupations[ik][n] * basis.kweights[ik]
            δρ[:, :, :, kpt.spin] .+= occ_weight .* (conj.(ψr) .* δψpr .+ ψr .* conj.(δψmr))

            if iszero(z)
                push!(static_diags, OrbitalDecomposition(
                    ik, n, 0.0, 0.0, norm(δψ_stern_p), norm(δψ_plus),0.0,0.0,res_p,0.0,0.0,
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
            delta_psi=δψ_plus_all,
            orbital_residual=max_res_plus,
            branch_residuals=(plus=max_res_plus, minus=max_res_minus))
end

function dynamic_sternheimer_response(scfres, δV::AbstractArray, z::Number;
                                      occupation_threshold=nothing)
    if iszero(z) || iszero(imag(z))
        return dynamic_sternheimer_response_fixed(scfres, δV, ComplexF64(0.0);
                                                  occupation_threshold=occupation_threshold)
    else
        return dynamic_sternheimer_response_fixed(scfres, δV, z;
                                                  occupation_threshold=occupation_threshold)
    end
end

end
