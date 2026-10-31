using LinearAlgebra
using Printf
using DFTK
include(joinpath(@__DIR__, "..", "src", "dftk", "si_scf.jl"))
include(joinpath(@__DIR__, "..", "src", "dftk", "si_response_adapter.jl"))
using .ShiftSchurDFTKSiSCF
using .ShiftSchurDFTKSiResponse

payload = scf_payload(run_si_scf())
static_smoke = dftk_static_response_smoke(payload)
δV = static_smoke.deltaV
reference = static_smoke.raw  # DFTK χ0 result

basis = payload.basis
ham = payload.raw.ham
ψ = payload.raw.ψ
ε = payload.raw.eigenvalues
occupations = [DFTK.to_cpu(occ) for occ in payload.raw.occupation]
threshold = DFTK.default_occupation_threshold(eltype(δV))
mask_occ, mask_empty = DFTK.occupied_empty_masks(occupations, threshold)

potential = DFTK.symmetrize_ρ(basis, δV)
rhs_all = DFTK.apply_potential_blochwave(basis, ψ, potential)

println("=== Per-k comparison of our Q*(H-ε)*Q residual vs DFTK δψ ===")
for ik in 1:length(basis.kpoints)
    ψk = ψ[ik]
    εk = ε[ik]
    kpt = basis.kpoints[ik]
    n_pw, n_bands = size(ψk)
    occ_mask = mask_occ[ik]
    # _to_indices logic
    occ_idx = occ_mask isa AbstractVector{Bool} ? findall(occ_mask) : 
              (eltype(occ_mask)<:Integer && length(occ_mask)==n_bands && all(x->x==0||x==1, occ_mask) ? [i for (i,v) in enumerate(occ_mask) if v!=0] : collect(Int, occ_mask))
    extra_mask = mask_empty[ik]
    extra_idx = extra_mask isa AbstractVector{Bool} ? findall(extra_mask) :
                (eltype(extra_mask)<:Integer && length(extra_mask)==n_bands && all(x->x==0||x==1, extra_mask) ? [i for (i,v) in enumerate(extra_mask) if v!=0] : collect(Int, extra_mask))

    P_all = ψk * ψk'
    Q_mat = Matrix{ComplexF64}(I - P_all)
    P_mat = Matrix{ComplexF64}(P_all)
    ham_block = ham.blocks[ik]

    for n in occ_idx[1:min(2,length(occ_idx))]
        εn = εk[n]
        rhs_n = rhs_all[ik][:, n]
        rhs_Q = Q_mat * rhs_n

        # Our dense A and solution
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
        δψ_ours = A_dense \ (-rhs_Q)

        # DFTK reference δψ
        δψ_dftk = reference.δψ[ik][:, n]

        # Residuals of both in OUR operator
        H_ours = ham_block * δψ_ours
        res_ours = norm(Q_mat * (H_ours .- εn.*δψ_ours .+ rhs_n)) / max(norm(rhs_Q), eps())
        H_dftk = ham_block * δψ_dftk
        res_dftk_in_ours = norm(Q_mat * (H_dftk .- εn.*δψ_dftk .+ rhs_n)) / max(norm(rhs_Q), eps())

        # Overlap with P_all and P_occ
        ψocc = ψk[:, occ_idx]
        P_occ = ψocc * ψocc'
        orth_ours_occ = norm(P_occ' * δψ_ours)
        orth_dftk_occ = norm(P_occ' * δψ_dftk)
        orth_ours_all = norm(P_mat' * δψ_ours)
        orth_dftk_all = norm(P_mat' * δψ_dftk)

        @printf("k=%d band=%d ε=%.4f ||rhs||=%.4e ||Qrhs||=%.4e ||ours||=%.4e ||dftk||=%.4e diff=%.4e res_ours=%.2e res_dftk_in_ours=%.2e orth_occ ours=%.2e dftk=%.2e orth_all ours=%.2e dftk=%.2e\n",
                ik, n, εn, norm(rhs_n), norm(rhs_Q), norm(δψ_ours), norm(δψ_dftk), norm(δψ_ours-δψ_dftk), res_ours, res_dftk_in_ours, orth_ours_occ, orth_dftk_occ, orth_ours_all, orth_dftk_all)
    end
end

println("Done")
