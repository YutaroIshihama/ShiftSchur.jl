"""Macroscopic dielectric matrix via dynamic Sternheimer response."""
module ShiftSchurDFTKDielectricMacroSternheimer

using LinearAlgebra
import DFTK
using LinearMaps
using IterativeSolvers

export epsilon_macro_sternheimer
export build_G_list
export coefficients_to_real, real_to_coefficients
export build_psi_r, build_psi_G

"""
    build_psi_r(basis, kpt, psi_G)

Transform plane-wave coefficients to real space using DFTK's physically
normalized inverse FFT. With the DFTK coefficient convention,
`sum(abs2, psi_r) * dV == sum(abs2, psi_G)`; normalized SCF orbitals
therefore satisfy `∫|ψ(r)|² dr = 1`. DFTK handles FFT normalization internally,
so no manual `sqrt(N)` factor or per-orbital renormalization is needed.
"""
function build_psi_r(basis, kpt, psi_G::AbstractVector)
    length(psi_G) == length(kpt.G_vectors) ||
        throw(DimensionMismatch("coefficient vector and G-vector list have different lengths"))
    return DFTK.ifft(basis, kpt, psi_G)
end

"""Transform a real-space field to the k-point's plane-wave coefficients."""
function build_psi_G(basis, kpt, field_r::AbstractArray)
    size(field_r) == Tuple(basis.fft_size) ||
        throw(DimensionMismatch("real-space field has the wrong FFT-grid shape"))
    return DFTK.fft(basis, kpt, field_r)
end

# Retain the existing names as aliases for callers of this module.
coefficients_to_real(basis, kpt, coefficients::AbstractVector) =
    build_psi_r(basis, kpt, coefficients)
real_to_coefficients(basis, kpt, field_r::AbstractArray) =
    build_psi_G(basis, kpt, field_r)

"""Build `exp(i q·r)` on the real-space grid, where
`r = lattice * [ix/nx, iy/ny, iz/nz]`."""
function build_phase_q(basis, q_red)
    nx, ny, nz = basis.fft_size
    phase = zeros(ComplexF64, nx, ny, nz)
    for ix in 0:nx-1, iy in 0:ny-1, iz in 0:nz-1
        phase[ix+1, iy+1, iz+1] = exp(2pi * im *
            (q_red[1]*ix/nx + q_red[2]*iy/ny + q_red[3]*iz/nz))
    end
    return phase
end

function build_G_list(basis, q_red; Ecut_chi=10.0)
    length(q_red) == 3 || throw(DimensionMismatch("q_red must have three components"))
    Ecut_chi > 0 || throw(ArgumentError("Ecut_chi must be positive"))

    q = Float64[x for x in q_red]
    recip = 2pi * inv(basis.model.lattice)'
    candidates = Set{NTuple{3, Int}}()
    push!(candidates, (0, 0, 0))
    for G in basis.kpoints[1].G_vectors
        push!(candidates, ntuple(i -> Int(G[i]), 3))
    end
    for gx in -2:2, gy in -2:2, gz in -2:2
        push!(candidates, (gx, gy, gz))
    end

    G_and_norms = Tuple{NTuple{3, Int}, Float64}[]
    for G in candidates
        qG_norm = norm(recip * (q .+ collect(G)))
        (G == (0, 0, 0) || qG_norm^2 / 2 < Ecut_chi) &&
            push!(G_and_norms, (G, qG_norm))
    end
    sort!(G_and_norms;
          by=x -> (x[1] == (0, 0, 0) ? 0 : 1, x[2], x[1][1], x[1][2], x[1][3]))
    return [Float64[G[1], G[2], G[3]] for (G, _) in Iterators.take(G_and_norms, 15)]
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
                                    eta=0.05, q_red=[0.01, 0, 0],
                                    Ecut_chi=10.0, nG=nothing)
    q_red = Float64[x for x in q_red]
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
    G_list = build_G_list(basis, q_red; Ecut_chi=Ecut_chi)
    if !isnothing(nG)
        nG isa Integer && !(nG isa Bool) ||
            throw(ArgumentError("nG must be a positive integer"))
        nG > 0 || throw(ArgumentError("nG must be positive"))
        G_list = G_list[1:min(nG, length(G_list))]
    end
    nG_actual = length(G_list)
    qG_norms = [norm(recip * (q_red .+ G)) for G in G_list]
    phase_qG_list = [build_phase_q(basis, q_red .+ G) for G in G_list]
    phase_minus_qG_list = [conj.(phase) for phase in phase_qG_list]
    @info "G_list size = $nG_actual, |q+G| = $qG_norms"
    @info "Macro Sternheimer: q_red=$q_red q_norm=$q_norm Omega=$Omega"

    vc_G = [4pi / (qG_norms[i]^2 + 1e-12) for i in eachindex(G_list)]
    vc = Diagonal(vc_G)
    chi0_mat_list = Matrix{ComplexF64}[]
    epsilon_mat_list = Matrix{ComplexF64}[]
    epsilon_M_LF = ComplexF64[]
    epsilon_M_NLF = ComplexF64[]
    chi0_00 = ComplexF64[]

    kpoint_data = NamedTuple[]
    for ik in eachindex(basis.kpoints)
        kpt = basis.kpoints[ik]
        occk = occupation[ik]
        occ_idx = findall(x -> x > 0.1, occk)
        isempty(occ_idx) && continue
        psi_occ = psi[ik][:, occ_idx]
        psi_occ_r = [build_psi_r(basis, kpt, psi[ik][:, n]) for n in occ_idx]
        push!(kpoint_data, (kpt=kpt, Hk=ham.blocks[ik], epsk=eigenvalues[ik],
                            occk=occk, occ_idx=occ_idx, psi_occ=psi_occ,
                            psi_occ_r=psi_occ_r, wk=kweights[ik]))
    end

    for omega in omegas
        z = omega + im*eta
        chi0_mat = zeros(ComplexF64, nG_actual, nG_actual)
        for (jGprime, Gprime) in enumerate(G_list)
            phase_qGprime = phase_qG_list[jGprime]
            delta_rho_r_Gprime = zeros(ComplexF64, fft_size...)
            for data in kpoint_data
                for (iocc, n) in enumerate(data.occ_idx)
                    fn = data.occk[n]
                    psi_n_r = data.psi_occ_r[iocc]
                    rhs_r = phase_qGprime .* psi_n_r
                    rhs_G = build_psi_G(basis, data.kpt, rhs_r)
                    x = solve_dynamic_gmres(data.Hk, data.psi_occ, data.epsk[n], rhs_G, z)
                    delta_psi_r = build_psi_r(basis, data.kpt, x)
                    delta_rho_r_Gprime .+= data.wk * fn .* conj.(psi_n_r) .* delta_psi_r
                end
            end
            for (iG, G) in enumerate(G_list)
                chi0_mat[iG, jGprime] =
                    sum(phase_minus_qG_list[iG] .* delta_rho_r_Gprime) * dV / Omega
            end
        end

        epsilon_mat = Matrix{ComplexF64}(I, nG_actual, nG_actual) - vc * chi0_mat
        epsilon_inv = inv(epsilon_mat)
        push!(chi0_mat_list, chi0_mat)
        push!(epsilon_mat_list, epsilon_mat)
        push!(chi0_00, chi0_mat[1, 1])
        push!(epsilon_M_LF, inv(epsilon_inv[1, 1]))
        push!(epsilon_M_NLF, epsilon_mat[1, 1])
        @info "omega=$omega chi0_00=$(chi0_mat[1, 1]) epsilon_M_LF=$(epsilon_M_LF[end]) epsilon_M_NLF=$(epsilon_M_NLF[end])"
    end
    return (omegas=omegas, G_list=G_list, chi0_mat_list=chi0_mat_list,
            epsilon_mat_list=epsilon_mat_list, epsilon_M_LF=epsilon_M_LF,
            epsilon_M_NLF=epsilon_M_NLF, chi0_00=chi0_00, vc=vc)
end

end
