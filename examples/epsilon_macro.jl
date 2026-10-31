using ShiftSchur
include(joinpath(@__DIR__, "..", "src", "dftk", "si_scf.jl"))
include(joinpath(@__DIR__, "..", "src", "dftk", "epsilon_macro.jl"))

using .ShiftSchurDFTKSiSCF
using .ShiftSchurDFTKDielectricMacro

println("Running Si SCF...")
payload = scf_payload(run_si_scf())

omegas = 0.0:0.05:1.0
eta = 0.05
q_red = [0.01, 0.0, 0.0]

println("Computing macroscopic epsilon_M for omegas=$omegas q=$q_red")
result = epsilon_macro(payload.raw, omegas; eta=eta, q_red=q_red)

output = joinpath(@__DIR__, "..", "docs", "epsilon_macro_spectrum.csv")
open(output, "w") do io
    println(io, "omega,eta,Re_chi0_00,Im_chi0_00,Re_epsM,Im_epsM")
    for (i, omega) in enumerate(result.omegas)
        chi0 = result.chi0[i]
        eps = result.epsilon[i]
        println(io, "$omega,$eta,$(real(chi0)),$(imag(chi0)),$(real(eps)),$(imag(eps))")
    end
end
println("Saved macro epsilon to $output")
for (i, omega) in enumerate(result.omegas)
    println("omega=$omega epsM=$(result.epsilon[i])")
end
