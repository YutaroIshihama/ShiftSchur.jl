"""
Macroscopic epsilon_M via dynamic Sternheimer (head only).

This revision removes vector-by-vector real-space renormalization.  The
plane-wave <-> real-space transforms use one fixed normalization factor,
so the amplitude of a Sternheimer solution is preserved.
"""
module ShiftSchurDFTKDielectricMacroSternheimer

using LinearAlgebra
import DFTK
import FFTW
using LinearMaps
using IterativeSolvers

export epsilon_macro_sternheimer
export coefficients_to_real, real_to_coefficients

"""Return the fixed coefficient-to-grid scale that preserves the L2 norm.

With Julia's FFT convention, `ifft` contains `1/N`.  Therefore

    psi_r = ifft(grid) * sqrt(N / dV)

satisfies `sum(abs2, psi_r) * dV == sum(abs2, coefficients)` up to roundoff.
The factor depends only on the basis/grid, never on the input vector.
"""
function _pw_to_real_scale(basis)
    N = prod(basis.fft_size)
    Omega = abs(det(basis.model.lattice))
    dV = Omega / N
    return sqrt(N / dV)
end

"""Map plane-wave coefficients to the FFT grid without renormalizing them."""
function coefficients_to_real(basis, kpt, coefficients::AbstractVector)
    fft_size = basis.fft_size
    grid = zeros(ComplexF64, fft_size...)
    length(coefficients) == length(kpt.G_vectors) ||
        throw(DimensionMismatch("coefficient vector and G-vector list have different lengths"))
    for (i, G) in enumerate(kpt.G_vectors)
        idx = ntuple(a -> mod(Int(G[a]), fft_size[a]) + 1, 3)
        grid[idx...] = coefficients[i]
    end
    return FFTW.ifft(grid) * _pw_to_real_scale(basis)
end

"""Map an FFT-grid field back to the selected plane-wave coefficients."""
function real_to_coefficients(basis, kpt, field_r::AbstractArray)
    size(field_r) == Tuple(basis.fft_size) ||
        throw(DimensionMismatch("real-space field has the wrong FFT-grid shape"))
    fft_size = basis.fft_size
    grid = FFTW.fft(field_r) / _pw_to_real_scale(basis)
    coefficients = zeros(ComplexF64, length(kpt.G_vectors))
    for (i, G) in enumerate(kpt.G_vectors)
        idx = ntuple(a -> mod(Int(G[a]), fft_size[a]) + 1, 3)
        coefficients[i] = grid[idx...]
    end
    return coefficients
end

function build_phase_q(basis, q_red)
    nx, ny, nz = basis.fft_size
    phase = zeros(ComplexF64, nx, ny, nz)
    for ix in 0:nx-1, iy in 0:ny-1, iz in 0:nz-1
        phase[ix+1, iy+1, iz+1] = exp(2pi * im *
            (q_red[1]*ix/nx + q_red[2]*iy/ny + q_red[3]*iz/nz))
    end
    return phase
end

function solve_dynamic_gmres(Hk, psi_occ, eps_n, rhs_n, z::Number;
                             reltol=1e-10, abstol=1e-12,
                             restart=80, maxiter=400)
    n_pw = size(psi_occ, 1)
    n_occ = size(psi_occ, 2)
    px = zeros(ComplexF64, n_pw)
    qx = similar(px)
    hqx = similar(px)
    tmp = similar(px)
    ptmp = similar(px)
    coeff = zeros(ComplexF64, n_occ)

    function project_P!(y, x)
        mul!(coeff, adjoint(psi_occ), x)
        mul!(y, psi_occ, coeff)
        return y
    end

    function apply_A!(y, x)
        project_P!(px, x)
        @. qx = x - px
        mul!(hqx, Hk, qx)
        @. tmp = hqx - (eps_n + z) * qx
        project_P!(ptmp, tmp)
        @. y = tmp - ptmp + px
        return y
    end

    project_P!(px, rhs_n)
    b = ComplexF64.(-rhs_n .+ px)
    bnorm = norm(b)
    bnorm < 1e-14 && return zeros(ComplexF64, n_pw)

    A = LinearMap{ComplexF64}(apply_A!, n_pw, n_pw; ismutating=true)
    x = zeros(ComplexF64, n_pw)
    x, history = gmres!(x, A, b;
                        restart=min(restart, n_pw), maxiter=maxiter,
                        reltol=reltol, abstol=abstol, log=true)

    Ax = similar(x)
    apply_A!(Ax, x)
    relres = norm(Ax - b) / bnorm
    project_P!(px, x)
    ortherr = norm(px) / max(norm(x), eps(Float64))
    target = max(reltol, abstol / bnorm)
    relres <= 10 * target ||
        error("dynamic GMRES failed: relres=$relres target=$target z=$z")
    ortherr <= 1e-8 ||
        error("dynamic GMRES left Q space: ortherr=$ortherr z=$z")
    @debug "DYNAMIC_GMRES" z relres ortherr history
    return x .- px
end

function epsilon_macro_sternheimer(scfres, omegas::AbstractVector;
                                    eta=0.05, q_red=[0.01, 0, 0])
    basis = scfres.basis
    ham = scfres.ham
    psi = scfres.ψ
    eigenvalues = scfres.eigenvalues
    occupation = [DFTK.to_cpu(o) for o in scfres.occupation]
    kweights = basis.kweights
    Omega = abs(det(basis.model.lattice))
    fft_size = basis.fft_size
    dV = Omega / prod(fft_size)
    recip = 2pi * inv(basis.model.lattice)'
    q_cart = recip * q_red
    q_norm = norm(q_cart)
    vc_q = 4pi / (q_norm^2 + 1e-12)
    @info "Macro Sternheimer: q_red=$q_red q_norm=$q_norm vc_q=$vc_q Omega=$Omega"
    phase_q = build_phase_q(basis, q_red)
    phase_minus_q = conj.(phase_q)
    epsM_list = ComplexF64[]
    chi0_list = ComplexF64[]
    for omega in omegas
        z = omega + im*eta
        delta_rho_r = zeros(ComplexF64, fft_size...)
        for ik in eachindex(basis.kpoints)
            kpt = basis.kpoints[ik]
            Hk = ham.blocks[ik]
            epsk = eigenvalues[ik]
            occk = occupation[ik]
            wk = kweights[ik]
            occ_idx = findall(x -> x > 0.1, occk)
            isempty(occ_idx) && continue
            psi_occ = psi[ik][:, occ_idx]
            for n in occ_idx
                fn = occk[n]
                psi_n_r = coefficients_to_real(basis, kpt, psi[ik][:, n])
                rhs_r = phase_q .* psi_n_r
                rhs_G = real_to_coefficients(basis, kpt, rhs_r)
                x = solve_dynamic_gmres(Hk, psi_occ, epsk[n], rhs_G, z)
                delta_psi_r = coefficients_to_real(basis, kpt, x)
                delta_rho_r .+= wk * fn .* conj.(psi_n_r) .* delta_psi_r
            end
        end
        chi0_00 = sum(phase_minus_q .* delta_rho_r) * dV / Omega
        epsM = 1 - vc_q * chi0_00
        push!(chi0_list, chi0_00)
        push!(epsM_list, epsM)
        @info "omega=$omega chi0_00=$chi0_00 epsM=$epsM"
    end
    return (omegas=omegas, chi0=chi0_list, epsilon=epsM_list, vc=vc_q)
end

end
