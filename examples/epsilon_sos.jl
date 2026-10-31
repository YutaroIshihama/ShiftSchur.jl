using ShiftSchur
include(joinpath(@__DIR__, "..", "src", "dftk", "si_scf.jl"))
include(joinpath(@__DIR__, "..", "src", "dftk", "si_response_adapter.jl"))
include(joinpath(@__DIR__, "..", "src", "dftk", "epsilon_sos.jl"))

using .ShiftSchurDFTKSiSCF
using .ShiftSchurDFTKDielectricSoS

println("Running Si SCF...")
payload = scf_payload(run_si_scf())

omegas = 0.0:0.05:1.0
eta = 0.05

println("Computing epsilon via SoS for omegas=$omegas")
result = epsilon_sos(payload.raw, omegas; eta=eta)

output = joinpath(@__DIR__, "..", "docs", "epsilon_sos_spectrum.csv")
open(output, "w") do io
    println(io, "omega,eta,Re_chi0,Im_chi0,Re_epsilon,Im_epsilon")
    for (i, omega) in enumerate(result.omegas)
        chi0 = result.chi0[i]
        eps = result.epsilon[i]
        println(io, "$omega,$eta,$(real(chi0)),$(imag(chi0)),$(real(eps)),$(imag(eps))")
    end
end
println("Saved SoS epsilon to $output")
for (i, omega) in enumerate(result.omegas)
    println("omega=$omega chi0=$(result.chi0[i]) epsilon=$(result.epsilon[i])")
end
