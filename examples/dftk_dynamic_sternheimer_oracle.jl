using LinearAlgebra
using Printf
using ShiftSchur
include(joinpath(@__DIR__, "..", "src", "dftk", "si_scf.jl"))
include(joinpath(@__DIR__, "..", "src", "dftk", "si_response_adapter.jl"))
using .ShiftSchurDFTKSiSCF
using .ShiftSchurDFTKSiResponse

payload = scf_payload(run_si_scf())
static_smoke = dftk_static_response_smoke(payload)
δV = static_smoke.deltaV
static_reference = static_smoke.delta_rho0

z_static = 1e-8im
static_limit = dynamic_sternheimer_response(payload.raw, δV, z_static)
static_relative_error = norm(static_limit.δρ - static_reference) /
                        max(norm(static_reference), eps(Float64))

println("=== DFTK dynamic Sternheimer oracle (validation only) ===")
println("static-limit z = ", z_static)
@printf("relative density error vs DFTK.apply_χ0 = %.8e\n", static_relative_error)
@printf("orbital Sternheimer residual           = %.8e\n", static_limit.orbital_residual)
@printf("occupied-space orthogonality error     = %.8e\n",
        static_limit.occupied_orthogonality_error)
@assert static_relative_error < 1e-5
@assert static_limit.orbital_residual < 1e-9
@assert static_limit.occupied_orthogonality_error < 1e-10

rows = NamedTuple[]
for (ω, η) in ((0.00, 0.05), (0.15, 0.05), (0.30, 0.05))
    response = dynamic_sternheimer_response(payload.raw, δV, ω + im * η)
    push!(rows, (omega=ω, eta=η, response_norm=norm(response.δρ),
                 orbital_residual=response.orbital_residual,
                 plus_residual=response.branch_residuals.plus,
                 minus_residual=response.branch_residuals.minus,
                 orthogonality_error=response.occupied_orthogonality_error))
    @printf("ω=%.2f η=%.2f norm(δρ)=%.8e residual=%.8e orthogonality=%.8e\n",
            ω, η, rows[end].response_norm, response.orbital_residual,
            response.occupied_orthogonality_error)
end

output = joinpath(@__DIR__, "..", "docs", "dftk_dynamic_sternheimer_oracle.csv")
open(output, "w") do io
    println(io, "omega,eta,response_norm,orbital_residual,plus_residual,minus_residual,orthogonality_error")
    println(io, "0.0,$(imag(z_static)),$(norm(static_limit.δρ))," *
               "$(static_limit.orbital_residual),$(static_limit.branch_residuals.plus)," *
               "$(static_limit.branch_residuals.minus),$(static_limit.occupied_orthogonality_error)")
    for row in rows
        println(io, "$(row.omega),$(row.eta),$(row.response_norm),$(row.orbital_residual)," *
                   "$(row.plus_residual),$(row.minus_residual),$(row.orthogonality_error)")
    end
end
println("Saved response norms and residuals to $output")
println("This oracle output is not a physical dielectric spectrum.")
