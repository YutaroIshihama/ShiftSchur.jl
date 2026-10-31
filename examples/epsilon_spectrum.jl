using ShiftSchur
include(joinpath(@__DIR__, "..", "src", "dftk", "si_scf.jl"))
include(joinpath(@__DIR__, "..", "src", "dftk", "si_response_adapter.jl"))
include(joinpath(@__DIR__, "..", "src", "dftk", "dynamic_sternheimer.jl"))
include(joinpath(@__DIR__, "..", "src", "dftk", "epsilon.jl"))

using .ShiftSchurDFTKSiSCF
using .ShiftSchurDFTKSiResponse
using .ShiftSchurDFTKDynamicSternheimer
using .ShiftSchurDFTKDielectric

println("Running Si SCF...")
payload = scf_payload(run_si_scf())
smoke = dftk_static_response_smoke(payload)
deltaV = smoke.deltaV

omegas = 0.0:0.05:1.0
eta = 0.05

println("Computing epsilon(omega) for omegas=$omegas eta=$eta")
result = epsilon_of_omega(payload.raw, deltaV, omegas; eta=eta)

# Save CSV
output = joinpath(@__DIR__, "..", "docs", "epsilon_spectrum.csv")
open(output, "w") do io
    println(io, "omega,eta,Re_chi0,Im_chi0,Re_epsilon,Im_epsilon,abs_epsilon")
    for (i, omega) in enumerate(result.omegas)
        chi0 = result.chi0[i]
        eps = result.epsilon[i]
        println(io, "$omega,$eta,$(real(chi0)),$(imag(chi0)),$(real(eps)),$(imag(eps)),$(abs(eps))")
    end
end
println("Saved epsilon spectrum to $output")

# Also print for quick check
for (i, omega) in enumerate(result.omegas)
    println("omega=$omega chi0=$(result.chi0[i]) epsilon=$(result.epsilon[i])")
end
