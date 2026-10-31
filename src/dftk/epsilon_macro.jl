"""
Macroscopic epsilon_M with |M|^2 and DeltaE stats
"""

module ShiftSchurDFTKDielectricMacro

using LinearAlgebra
import DFTK
import FFTW

export epsilon_macro

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

function epsilon_macro(scfres, omegas::AbstractVector; eta=0.05, q_red=[0.01,0,0])
    basis = scfres.basis
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
    @info "Macro q_red=$q_red q_norm=$q_norm vc_q=$vc_q Omega=$Omega"

    phase_q = build_phase_q(basis, q_red)

    chi0_00_list = ComplexF64[]
    epsM_list = ComplexF64[]

    for omega in omegas
        z = omega + im*eta
        chi0_00 = 0.0im
        maxM2 = 0.0
        meanM2 = 0.0
        count = 0

        for ik in 1:length(basis.kpoints)
            kpt = basis.kpoints[ik]
            epsk = eigenvalues[ik]
            occk = occupation[ik]
            wk = kweights[ik]

            occ_idx = findall(x -> x > 0.1, occk)
            emp_idx = findall(x -> x < 0.1, occk)
            isempty(occ_idx) && continue
            isempty(emp_idx) && continue

            for n in occ_idx, m in emp_idx
                fn = occk[n]; fm = occk[m]
                abs(fn - fm) < 1e-10 && continue
                # Build psi_r for n,m
                psi_n_r = build_psi_r(basis, kpt, psi[ik][:,n])
                psi_m_r = build_psi_r(basis, kpt, psi[ik][:,m])
                M = sum(conj.(psi_n_r) .* phase_q .* psi_m_r) * dV
                M2 = abs2(M)
                dE = epsk[m] - epsk[n]
                # Stats for first omega only
                if omega == omegas[1] && ik <= 2 && n <= 2 && m <= 8
                    @info "STAT k=$ik n=$n(m=$(epsk[n])) m=$m(E=$(epsk[m])) dE=$dE |M|^2=$M2 M=$M"
                end
                maxM2 = max(maxM2, M2)
                meanM2 += M2
                count += 1

                denom = epsk[n] - epsk[m] + z
                chi0_00 += wk * (fn - fm) * M2 / denom
            end
        end
        meanM2 /= max(count,1)
        chi0_00 = chi0_00 / Omega
        eps_00 = 1 - vc_q * chi0_00
        epsM = eps_00
        push!(chi0_00_list, chi0_00)
        push!(epsM_list, epsM)
        @info "RESULT omega=$omega chi0_00=$chi0_00 epsM=$epsM maxM2=$maxM2 meanM2=$meanM2 count=$count"
    end

    return (omegas=omegas, chi0=chi0_00_list, epsilon=epsM_list, vc=vc_q)
end

end
