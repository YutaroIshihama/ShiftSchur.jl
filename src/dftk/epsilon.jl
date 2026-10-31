"""
epsilon(omega) skeleton from dynamic chi0
"""

module ShiftSchurDFTKDielectric

using LinearAlgebra
import DFTK
import FFTW
using Statistics: mean

export epsilon_of_omega

function compute_chi0_GG(basis, delta_rho, deltaV)
    rho = mean(delta_rho, dims=4)
    rho = dropdims(rho, dims=4)
    V = mean(deltaV, dims=4)
    V = dropdims(V, dims=4)
    rho_G = FFTW.fft(rho)
    V_G = FFTW.fft(V)
    nx, ny, nz = size(rho)
    idx_G0 = (2,1,1)
    idx_minusG0 = (nx,1,1)
    chi0 = (rho_G[idx_G0...] + rho_G[idx_minusG0...]) / (V_G[idx_G0...] + V_G[idx_minusG0...] + eps())
    return chi0
end

function coulomb_kernel_G0(basis)
    lattice = basis.model.lattice
    recip = 2π * inv(lattice)'
    G0_cart = recip[:,1]
    G0_norm = norm(G0_cart)
    vc = 4π / (G0_norm^2 + 1e-12)
    return vc, G0_norm
end

function epsilon_of_omega(scfres, deltaV, omegas::AbstractVector; eta=0.05)
    basis = scfres.basis
    vc, G0_norm = coulomb_kernel_G0(basis)
    @info "epsilon skeleton: vc(G0)=$vc, |G0|=$G0_norm, G0=(1,0,0)"
    epsilons = ComplexF64[]
    chi0s = ComplexF64[]
    for omega in omegas
        z = omega + im*eta
        resp = Main.ShiftSchurDFTKDynamicSternheimer.dynamic_sternheimer_response_fixed(scfres, deltaV, z)
        delta_rho = resp.delta_rho !== nothing ? resp.delta_rho : resp.δρ
        chi0 = compute_chi0_GG(basis, delta_rho, deltaV)
        epsilon = 1 - vc * chi0
        push!(chi0s, chi0)
        push!(epsilons, epsilon)
        @info "omega=$omega eta=$eta chi0=$chi0 epsilon=$epsilon |epsilon|=$(abs(epsilon))"
    end
    return (omegas=omegas, chi0=chi0s, epsilon=epsilons, vc=vc, G0_norm=G0_norm)
end

end
