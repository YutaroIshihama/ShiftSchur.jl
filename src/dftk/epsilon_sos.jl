"""
epsilon via Sum-over-States (SoS) - G-space shift only, no G_to_r
"""

module ShiftSchurDFTKDielectricSoS

using LinearAlgebra
import DFTK

export epsilon_sos

function coulomb_kernel_G0(basis)
    lattice = basis.model.lattice
    recip = 2π * inv(lattice)'
    G0_cart = recip[:,1]
    G0_norm = norm(G0_cart)
    vc = 4π / (G0_norm^2 + 1e-12)
    return vc, G0_norm, G0_cart
end

function matrix_element_G0_Gspace(kpt, psi_n_G, psi_m_G)
    # <n|e^{iG0}|m> = Σ_G ψ_n^*(G) ψ_m(G - G0_red)
    # G0_red = (1,0,0)
    Gvecs = kpt.G_vectors
    g_to_idx = Dict{NTuple{3,Int}, Int}()
    for (i,g) in enumerate(Gvecs)
        # g is SVector{3,Int} or similar
        tup = (Int(g[1]), Int(g[2]), Int(g[3]))
        g_to_idx[tup] = i
    end
    overlap = 0.0im
    G0 = (1,0,0)
    for (i,G) in enumerate(Gvecs)
        Gt = (Int(G[1])-G0[1], Int(G[2])-G0[2], Int(G[3])-G0[3])
        j = get(g_to_idx, Gt, nothing)
        if j !== nothing
            overlap += conj(psi_n_G[i]) * psi_m_G[j]
        end
    end
    return overlap
end

function epsilon_sos(scfres, omegas::AbstractVector; eta=0.05)
    basis = scfres.basis
    psi = scfres.ψ
    eigenvalues = scfres.eigenvalues
    occupation = [DFTK.to_cpu(o) for o in scfres.occupation]
    kweights = basis.kweights
    vc, G0_norm, G0_cart = coulomb_kernel_G0(basis)
    Omega = abs(det(basis.model.lattice))

    @info "SoS epsilon: vc=$vc G0_norm=$G0_norm Omega=$Omega"

    chi0_list = ComplexF64[]
    eps_list = ComplexF64[]

    for omega in omegas
        z = omega + im*eta
        chi0 = 0.0im
        for ik in 1:length(basis.kpoints)
            kpt = basis.kpoints[ik]
            n_bands = size(psi[ik],2)
            epsk = eigenvalues[ik]
            occk = occupation[ik]
            wk = kweights[ik]

            # Precompute M
            M = zeros(ComplexF64, n_bands, n_bands)
            for n in 1:n_bands, m in 1:n_bands
                M[n,m] = matrix_element_G0_Gspace(kpt, psi[ik][:,n], psi[ik][:,m])
            end

            for n in 1:n_bands, m in 1:n_bands
                n==m && continue
                fn = occk[n]; fm = occk[m]
                abs(fn - fm) < 1e-10 && continue
                denom = epsk[n] - epsk[m] + z
                chi0 += wk * (fn - fm) * abs2(M[n,m]) / denom
            end
        end
        chi0 = chi0 / Omega
        epsilon = 1 - vc * chi0
        push!(chi0_list, chi0)
        push!(eps_list, epsilon)
        @info "SoS omega=$omega chi0=$chi0 epsilon=$epsilon"
    end

    return (omegas=omegas, chi0=chi0_list, epsilon=eps_list, vc=vc)
end

end
