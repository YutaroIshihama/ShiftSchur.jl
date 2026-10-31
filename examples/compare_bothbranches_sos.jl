using Printf
using ShiftSchur
include(joinpath(@__DIR__,"..","src","dftk","si_scf.jl"))
include(joinpath(@__DIR__,"..","src","dftk","epsilon_macro_bothbranches.jl"))
using .ShiftSchurDFTKSiSCF
using .ShiftSchurDFTKDielectricMacroBothBranches

println("Running Si SCF...")
payload=scf_payload(run_si_scf())
omegas=[0.0,0.1,0.2,0.3,0.4]
eta=0.05
q_red=[0.01,0.0,0.0]
result=compare_sternheimer_sos(payload.raw,omegas;eta,q_red)

output=joinpath(@__DIR__,"..","docs","epsilon_bothbranches_vs_sos.csv")
open(output,"w") do io
    println(io,"omega,Re_chi_stern,Im_chi_stern,Re_chi_sos,Im_chi_sos,Re_eps_stern,Im_eps_stern,Re_eps_sos,Im_eps_sos,rel_chi_err,rel_eps_err")
    for i in eachindex(result.omegas)
        cs=result.chi_sternheimer[i]; cq=result.chi_sos[i]
        es=result.epsilon_sternheimer[i]; eq=result.epsilon_sos[i]
        println(io,"$(result.omegas[i]),$(real(cs)),$(imag(cs)),$(real(cq)),$(imag(cq)),$(real(es)),$(imag(es)),$(real(eq)),$(imag(eq)),$(result.relative_chi_error[i]),$(result.relative_epsilon_error[i])")
    end
end
println("omega   Re eps(St)   Im eps(St)   Re eps(SoS)  Im eps(SoS)  rel chi     rel eps")
for i in eachindex(result.omegas)
    es=result.epsilon_sternheimer[i]; eq=result.epsilon_sos[i]
    @printf("%5.2f  %11.6f  %11.6f  %11.6f  %11.6f  %.3e  %.3e\n",result.omegas[i],real(es),imag(es),real(eq),imag(eq),result.relative_chi_error[i],result.relative_epsilon_error[i])
end
println("Saved comparison to $output")
println("NOTE: same-k finite-q diagnostic; k->k+q is still pending.")
