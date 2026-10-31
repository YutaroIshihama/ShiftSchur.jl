"""
Macroscopic epsilon_M via dynamic Sternheimer (no sum over conduction bands)
Uses v28 dense solver for (H - eps - z) Q delta_psi = -Q deltaV psi
deltaV = exp(i q·r), q small
delta_rho(ω) = Σ_n f_n psi_n^* delta_psi_n(ω)  (complex)
chi0_00(q,ω) = (1/Ω) ∫ e^{-iq·r} delta_rho(r) dr
epsilon_00 = 1 - v_c(q) chi0_00
epsilon_M ≈ epsilon_00 (head only, no local fields) or 1/epsilon^{-1}_00 with full matrix
"""

module ShiftSchurDFTKDielectricMacroSternheimer

using LinearAlgebra
import DFTK
import FFTW

export epsilon_macro_sternheimer

function build_psi_r(basis, kpt, psi_G)
    fft_size = basis.fft_size
    nx, ny, nz = fft_size
    grid = zeros(ComplexF64, nx, ny, nz)
    for (i,G) in enumerate(kpt.G_vectors)
        ix = mod(Int(G[1]), nx) + 1
        iy = mod(Int(G[2]), ny) + 1
        iz = mod(Int(G[3]), nz) + 1
        grid[ix, iy, iz] = psi_G[i]
    end
    psi_r = FFTW.ifft(grid) * sqrt(prod(fft_size))
    Omega = abs(det(basis.model.lattice))
    dV = Omega / prod(fft_size)
    norm2 = sum(abs2.(psi_r)) * dV
    psi_r ./= sqrt(norm2)
    return psi_r
end

function build_phase_q(basis, q_red)
    nx, ny, nz = basis.fft_size
    phase = zeros(ComplexF64, nx, ny, nz)
    for ix in 0:nx-1, iy in 0:ny-1, iz in 0:nz-1
        dot = q_red[1]*ix/nx + q_red[2]*iy/ny + q_red[3]*iz/nz
        phase[ix+1, iy+1, iz+1] = exp(2π*im * dot)
    end
    return phase
end

function dense_hamiltonian_matrix(Hk, n_pw::Int)
    Hmat = zeros(ComplexF64, n_pw, n_pw)
    e = zeros(ComplexF64, n_pw)
    for i in 1:n_pw
        fill!(e, 0); e[i]=1
        Hmat[:,i] .= Hk * e
    end
    return Hmat
end

function solve_dynamic_dense(Hk, psi_occ, eps_n, rhs_n, z::Number)
    n_pw = size(psi_occ, 1)
    P_occ = psi_occ * psi_occ'
    Q_occ = Matrix{ComplexF64}(I - P_occ)
    b = -Q_occ * rhs_n
    norm(b) < 1e-14 && return zeros(ComplexF64, n_pw)
    Hmat = dense_hamiltonian_matrix(Hk, n_pw)
    A = Q_occ * (Hmat - (eps_n + z)*I) * Q_occ
    A_reg = A + P_occ
    x = A_reg \ b
    x = Q_occ * x
    return x
end

function epsilon_macro_sternheimer(scfres, omegas::AbstractVector; eta=0.05, q_red=[0.01,0,0])
    basis = scfres.basis
    ham = scfres.ham
    psi = scfres.ψ
    eigenvalues = scfres.eigenvalues
    occupation = [DFTK.to_cpu(o) for o in scfres.occupation]
    kweights = basis.kweights
    Omega = abs(det(basis.model.lattice))
    fft_size = basis.fft_size
    dV = Omega / prod(fft_size)

    lattice = basis.model.lattice
    recip = 2π * inv(lattice)'
    q_cart = recip * q_red
    q_norm = norm(q_cart)
    vc_q = 4π / (q_norm^2 + 1e-12)
    @info "Macro Sternheimer: q_red=$q_red q_norm=$q_norm vc_q=$vc_q Omega=$Omega"

    phase_q = build_phase_q(basis, q_red)
    phase_minus_q = conj.(phase_q)

    epsM_list = ComplexF64[]
    chi0_list = ComplexF64[]

    for omega in omegas
        z = omega + im*eta
        # delta_rho accumulated over kpoints, on real grid
        delta_rho_r = zeros(ComplexF64, fft_size...)

        for ik in 1:length(basis.kpoints)
            kpt = basis.kpoints[ik]
            Hk = ham.blocks[ik]
            n_pw = size(psi[ik],1)
            n_bands = size(psi[ik],2)
            epsk = eigenvalues[ik]
            occk = occupation[ik]
            wk = kweights[ik]

            # Determine occupied indices (simple threshold)
            occ_idx = findall(x -> x > 0.1, occk)  # occupied
            isempty(occ_idx) && continue
            psi_occ = psi[ik][:, occ_idx]
            n_occ = length(occ_idx)

            # Build rhs for each occupied band: deltaV * psi_n
            # deltaV = e^{iq·r}, in real space multiply
            for (ii, n) in enumerate(occ_idx)
                fn = occk[n]
                psi_n_G = psi[ik][:, n]
                psi_n_r = build_psi_r(basis, kpt, psi_n_G)
                # rhs_r = e^{iq·r} * psi_n(r)
                rhs_r = phase_q .* psi_n_r
                # Convert rhs_r back to G space: need G_to_r inverse is r_to_G
                # For simplicity, we compute rhs_G via FFT: r_to_G
                # Place rhs_r on grid, FFT to get G coefficients, then project onto kpt basis
                # Use FFTW.fft
                rhs_grid = FFTW.fft(rhs_r) / sqrt(prod(fft_size))  # inverse of earlier ifft scaling
                # Now extract G components corresponding to kpt.G_vectors
                rhs_G = zeros(ComplexF64, n_pw)
                for (i,G) in enumerate(kpt.G_vectors)
                    ix = mod(Int(G[1]), fft_size[1]) + 1
                    iy = mod(Int(G[2]), fft_size[2]) + 1
                    iz = mod(Int(G[3]), fft_size[3]) + 1
                    rhs_G[i] = rhs_grid[ix, iy, iz]
                end
                # Normalize rhs_G to match psi normalization
                # The FFT scaling we used should make ∫|psi|^2=1, so rhs_G should be similarly scaled
                # Solve Sternheimer
                eps_n = epsk[n]
                x = solve_dynamic_dense(Hk, psi_occ, eps_n, rhs_G, z)
                # x is delta_psi_G
                # Convert delta_psi_G to real space
                delta_psi_r = build_psi_r(basis, kpt, x)
                # delta_rho contribution: f_n * psi_n^* * delta_psi
                delta_rho_r .+= wk * fn * conj.(psi_n_r) .* delta_psi_r
            end
        end

        # chi0_00 = (1/Ω) ∫ e^{-iq·r} delta_rho(r) dr
        chi0_00 = sum(phase_minus_q .* delta_rho_r) * dV / Omega
        eps_00 = 1 - vc_q * chi0_00
        epsM = eps_00  # head only, no local fields

        push!(chi0_list, chi0_00)
        push!(epsM_list, epsM)
        @info "omega=$omega chi0_00=$chi0_00 epsM=$epsM"
    end

    return (omegas=omegas, chi0=chi0_list, epsilon=epsM_list, vc=vc_q)
end

end
