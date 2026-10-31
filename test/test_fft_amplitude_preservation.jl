using Test
using LinearAlgebra
using Random
using DFTK

include(joinpath(@__DIR__, "..", "src", "dftk", "si_scf.jl"))
include(joinpath(@__DIR__, "..", "src", "dftk", "epsilon_macro_sternheimer.jl"))
using .ShiftSchurDFTKSiSCF
using .ShiftSchurDFTKDielectricMacroSternheimer

@testset "fixed-amplitude PW/real transforms" begin
    payload = scf_payload(run_si_scf())
    basis = payload.basis
    kpt = basis.kpoints[1]
    c = ComplexF64.(payload.psi[1][:, 1])
    N = prod(basis.fft_size)
    Omega = abs(det(basis.model.lattice))
    dV = Omega / N

    psi_r = coefficients_to_real(basis, kpt, c)
    @test sum(abs2, psi_r) * dV ≈ sum(abs2, c) rtol=2e-12 atol=2e-12
    @test real_to_coefficients(basis, kpt, psi_r) ≈ c rtol=2e-12 atol=2e-12

    Random.seed!(20261031)
    x = randn(ComplexF64, length(c))
    alpha = 0.037 - 0.021im
    x_r = coefficients_to_real(basis, kpt, x)
    alpha_x_r = coefficients_to_real(basis, kpt, alpha .* x)

    # This is the regression that fails if each delta_psi is normalized to norm 1.
    @test alpha_x_r ≈ alpha .* x_r rtol=2e-12 atol=2e-12
    @test sqrt(sum(abs2, x_r) * dV) ≈ norm(x) rtol=2e-12 atol=2e-12
    @test sqrt(sum(abs2, alpha_x_r) * dV) ≈ abs(alpha) * norm(x) rtol=2e-12 atol=2e-12
end
