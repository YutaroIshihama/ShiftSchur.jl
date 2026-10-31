using ShiftSchur
include(joinpath(@__DIR__,"..","src","dftk","si_scf.jl"))
include(joinpath(@__DIR__,"..","src","dftk","si_response_adapter.jl"))
using .ShiftSchurDFTKSiSCF, .ShiftSchurDFTKSiResponse
cfg=SiSCFConfig()
scf=run_si_scf(cfg); payload=scf_payload(scf)
p=build_si_response_problem(payload;q=(0.0,0.0,0.0),perturbation=:density_head)
validate_response_problem(p)
M,b=p.model,p.probe
eta=0.10
omegagrid=range(0.0,0.8;length=41)
shifts,hist=greedy_rational_shifts_aposteriori(M,omegagrid,eta,b;s_max=16,tol=1e-4)
Q=rational_krylov_basis(M.D,M.B',shifts)
RM=ReducedSchurModel(M,Q)
open(joinpath(@__DIR__,"si_shift_schur_response.csv"),"w") do io
    println(io,"omega_reduced,eta,exact_norm,reduced_norm,relative_action_error")
    for omega in omegagrid
        z=omega+im*eta
        exact=self_energy(M,z)*b
        red=apply_reduced_self_energy(RM,z,b)
        err=norm(exact-red)/max(norm(exact),eps())
        println(io,"$omega,$eta,$(norm(exact)),$(norm(red)),$err")
    end
end
println("DFTK-Si ShiftSchur OK: npoles=$(length(shifts)), k=$(size(Q,2)), final estimator=$(hist[end].max_err)")
