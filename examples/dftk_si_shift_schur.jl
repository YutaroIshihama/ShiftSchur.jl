using LinearAlgebra, ShiftSchur
include(joinpath(@__DIR__,"..","src","dftk","si_scf.jl"))
include(joinpath(@__DIR__,"..","src","dftk","si_response_adapter.jl"))
using .ShiftSchurDFTKSiSCF, .ShiftSchurDFTKSiResponse
scf=run_si_scf(); payload=scf_payload(scf)
smoke=dftk_static_response_smoke(payload)
println("DFTK static response smoke OK: ||delta_rho0|| = ",norm(smoke.delta_rho0))
p=build_si_response_problem(payload; q=(0.0,0.0,0.0), perturbation=:periodic_density_probe)
validate_response_problem(p)
println("Static DFTK sampled chi0 -> SchurModel OK")
println("  dimension = ",size(p.model.A,1))
println("  static fit relative error = ",p.metadata.static_fit_relative_error)
println("  NOTE: static_only=true; no physical dynamic spectrum is claimed yet.")
z=0.0+0.0im
sigma0=self_energy(p.model,z)
println("  Sigma(0) consistency residual = ",norm(sigma0 + p.metadata.chi0_matrix)/max(norm(p.metadata.chi0_matrix),eps()))
