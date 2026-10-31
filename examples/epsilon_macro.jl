using ShiftSchur
include(joinpath(@__DIR__, "..", "src", "dftk", "si_scf.jl"))
include(joinpath(@__DIR__, "..", "src", "dftk", "epsilon_macro_sternheimer.jl"))

using .ShiftSchurDFTKSiSCF
using .ShiftSchurDFTKDielectricMacroSternheimer

println("Running Si SCF...")
payload = scf_payload(run_si_scf())

omegas = [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]
eta = 0.05
q_red = [0.01, 0.0, 0.0]

println("Computing macroscopic epsilon_M via Sternheimer (no SoS) for omegas=$omegas q=$q_red")
result = epsilon_macro_sternheimer(payload.raw, omegas; eta=eta, q_red=q_red)

output = joinpath(@__DIR__, "..", "docs", "epsilon_macro_sternheimer.csv")
open(output, "w") do io
    println(io, "omega,Re_chi0,Im_chi0,Re_epsM,Im_epsM")
    for (i, omega) in enumerate(result.omegas)
        println(io, "$omega,$(real(result.chi0[i])),$(imag(result.chi0[i])),$(real(result.epsilon[i])),$(imag(result.epsilon[i]))")
    end
end
println("Saved macro Sternheimer epsilon to $output")
for (i, omega) in enumerate(result.omegas)
    println("omega=$omega chi0=$(result.chi0[i]) epsM=$(result.epsilon[i])")
end
