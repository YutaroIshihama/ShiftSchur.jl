"""
Macroscopic dielectric epsilon_M via small q real-space method
"""

module ShiftSchurDFTKDielectricMacro

using LinearAlgebra
import DFTK
import FFTW

export epsilon_macro

function coulomb_kernel_G0(basis)
    lattice = basis.model.lattice
    recip = 2π * inv(lattice)'
    G0_cart = recip[:,1]
    G0_norm = norm(G0_cart)
    vc = 4π / (G0_norm^2 + 1e-12)
    return vc, G0_norm, G0_cart
end

function build_psi_r(basis, kpt, psi_G)
    # Build real-space wavefunction from G coefficients
    fft_size = basis.fft_size
    nx, ny, nz = fft_size
    grid = zeros(ComplexF64, nx, ny, nz)
    Gvecs = kpt.G_vectors
    # Gvecs are in reduced coordinates (integers)
    for (i,G) in enumerate(Gvecs)
        gx = Int(G[1]); gy = Int(G[2]); gz = Int(G[3])
        ix = mod(gx, nx) + 1
        iy = mod(gy, ny) + 1
        iz = mod(gz, nz) + 1
        grid[ix, iy, iz] = psi_G[i]
    end
    # ifft: DFTK's convention is G_to_r = ifft with normalization sqrt(Ω)/N?
    # We'll use FFTW ifft and normalize to get ∫|ψ|^2 =1
    # ifft in FFTW is unnormalized: ifft = 1/N * Σ
    # We want ψ(r) such that sum |ψ(r)|^2 * Ω/N =1
    psi_r = FFTW.ifft(grid) * sqrt(prod(fft_size))  # heuristic scaling
    # Normalize: sum |psi_r|^2 * Ω/N =1 => |psi_r|^2 sum = N/Ω
    # Let's compute norm and rescale
    # For now, normalize explicitly
    Omega = abs(det(basis.model.lattice))
    dV = Omega / prod(fft_size)
    norm2 = sum(abs2.(psi_r)) * dV
    psi_r ./= sqrt(norm2)
    return psi_r
end

function build_phase_q(basis, q_red)
    # q_red in reduced coordinates, e^{i q·r} = exp(2π i (q_red· frac))
    # frac = (ix/nx, iy/ny, iz/nz)
    fft_size = basis.fft_size
    nx, ny, nz = fft_size
    phase = zeros(ComplexF64, nx, ny, nz)
    for ix in 0:nx-1, iy in 0:ny-1, iz in 0:nz-1
        # q·r = 2π * q_red·(frac) ??? Actually r = lattice * frac, q_cart = recip * q_red, q·r = 2π q_red·frac
        dot = q_red[1]*ix/nx + q_red[2]*iy/ny + q_red[3]*iz/nz
        phase[ix+1, iy+1, iz+1] = exp(2π*im * dot)
    end
    return phase
end

function matrix_element_q(basis, kpt, psi_n_G, psi_m_G, q_red)
    psi_n_r = build_psi_r(basis, kpt, psi_n_G)
    psi_m_r = build_psi_r(basis, kpt, psi_m_G)
    phase = build_phase_q(basis, q_red)
    # <n|e^{iq·r}|m> = ∫ ψ_n^* e^{iq·r} ψ_m d^3r
    Omega = abs(det(basis.model.lattice))
    dV = Omega / prod(basis.fft_size)
    overlap = sum(conj.(psi_n_r) .* phase .* psi_m_r) * dV
    return overlap
end

function epsilon_macro(scfres, omegas::AbstractVector; eta=0.05, q_red=[0.01,0,0])
    basis = scfres.basis
    psi = scfres.ψ
    eigenvalues = scfres.eigenvalues
    occupation = [DFTK.to_cpu(o) for o in scfres.occupation]
    kweights = basis.kweights
    Omega = abs(det(basis.model.lattice))

    chi0_00_list = ComplexF64[]
    epsM_list = ComplexF64[]

    for omega in omegas
        z = omega + im*eta
        chi0_00 = 0.0im
        for ik in 1:length(basis.kpoints)
            kpt = basis.kpoints[ik]
            n_bands = size(psi[ik],2)
            epsk = eigenvalues[ik]
            occk = occupation[ik]
            wk = kweights[ik]

            for n in 1:n_bands, m in 1:n_bands
                n==m && continue
                fn = occk[n]; fm = occk[m]
                abs(fn - fm) < 1e-10 && continue
                M = matrix_element_q(basis, kpt, psi[ik][:,n], psi[ik][:,m], q_red)
                denom = epsk[n] - epsk[m] + z
                chi0_00 += wk * (fn - fm) * abs2(M) / denom
            end
        end
        chi0_00 = chi0_00 / Omega
        # For small q, v_c(q) = 4π/|q+G|^2, G=0
        lattice = basis.model.lattice
        recip = 2π * inv(lattice)'
        q_cart = recip * q_red
        q_norm = norm(q_cart)
        vc_q = 4π / (q_norm^2 + 1e-12)
        # epsilon_{00} = 1 - vc_q * chi0_00
        eps_00 = 1 - vc_q * chi0_00
        # Macroscopic epsilon_M = 1 / (epsilon^{-1})_{00}
        # For single G=0, ignoring local fields, epsilon_M ≈ eps_00
        # With local fields, need full matrix inversion, but for small q head dominates
        epsM = eps_00  # approximation, without local fields
        # If we want with local fields, need full epsilon matrix, but for now head only
        push!(chi0_00_list, chi0_00)
        push!(epsM_list, epsM)
        @info "Macro omega=$omega q=$q_red chi0_00=$chi0_00 eps_00=$eps_00 epsM=$epsM"
    end

    return (omegas=omegas, chi0=chi0_00_list, epsilon=epsM_list)
end

end
