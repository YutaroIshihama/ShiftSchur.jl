using LinearAlgebra
using Printf
using ShiftSchur
using DFTK
include(joinpath(@__DIR__, "..", "src", "dftk", "si_scf.jl"))
include(joinpath(@__DIR__, "..", "src", "dftk", "si_response_adapter.jl"))
using .ShiftSchurDFTKSiSCF
using .ShiftSchurDFTKSiResponse

payload = scf_payload(run_si_scf())
static_smoke = dftk_static_response_smoke(payload)
δV = static_smoke.deltaV
reference_result = static_smoke.raw
static_reference = reference_result.δρ

z_static = 1e-8im
static_limit = dynamic_sternheimer_response(payload.raw, δV, z_static)
static_orbital_reference = dftk_static_orbital_response_reference(
    payload.raw, δV)
occupation_threshold = DFTK.default_occupation_threshold(eltype(δV))
occupations_cpu = [DFTK.to_cpu(occ) for occ in payload.raw.occupation]
occupied_masks, extra_masks = DFTK.occupied_empty_masks(
    occupations_cpu, occupation_threshold)
zero_delta_psi = zero.(reference_result.δψ)
zero_delta_occupation = zero.(reference_result.δoccupation)
reconstruct_density(δψ, δoccupation) = DFTK.compute_δρ(
    payload.basis, payload.psi, δψ, payload.raw.occupation, δoccupation;
    occupation_threshold=occupation_threshold)

dftk_reconstructed = reconstruct_density(reference_result.δψ,
                                           reference_result.δoccupation)
our_plus_with_dftk_occupation = reconstruct_density(
    static_limit.delta_psi_plus, reference_result.δoccupation)
our_plus_with_zero_occupation = reconstruct_density(
    static_limit.delta_psi_plus, zero_delta_occupation)
dftk_orbital_only = reconstruct_density(reference_result.δψ, zero_delta_occupation)
dftk_occupation_only = reconstruct_density(zero_delta_psi,
                                            reference_result.δoccupation)
missing_orbital_variation = map(-, reference_result.δψ,
                                static_limit.delta_psi_plus)
missing_orbital_density = reconstruct_density(missing_orbital_variation,
                                              zero_delta_occupation)
manual_dynamic = static_limit.δρ
relative_error(a, b) = norm(a - b) / max(norm(b), eps(Float64))
relative_orbital_error(a, b) =
    sqrt(sum(norm(a[ik] - b[ik])^2 for ik in eachindex(a))) /
    max(sqrt(sum(norm(bk)^2 for bk in b)), eps(Float64))
component_norm(x::Nothing) = nothing
component_norm(x::Number) = abs(x)
component_norm(x::AbstractArray{<:Number}) = norm(x)
function component_norm(x::AbstractArray)
    total = 0.0
    for value in x
        value_norm = component_norm(value)
        value_norm === nothing || (total += value_norm^2)
    end
    sqrt(total)
end
component_norm(::Any) = nothing

println("=== DFTK.apply_χ0 result fields ===")
for name in propertynames(reference_result)
    value = getproperty(reference_result, name)
    println("$name: type=$(typeof(value)), norm=$(component_norm(value))")
end

reference_density_norm = norm(static_reference)
reconstructions = (
    A_dftk_orbitals_and_occupations=dftk_reconstructed,
    B_our_plus_dftk_occupations=our_plus_with_dftk_occupation,
    C_our_plus_zero_occupations=our_plus_with_zero_occupation,
    D_manual_plus_minus=manual_dynamic,
    dftk_orbital_only=dftk_orbital_only,
    dftk_occupation_only=dftk_occupation_only,
    missing_orbital_delta_density=missing_orbital_density,
)

println("=== Static component-wise density reconstruction ===")
@printf("reference norm = %.8e\n", reference_density_norm)
println("n_spin_components = ", payload.basis.model.n_spin_components)
println("k-point weights = ", payload.basis.kweights)
println("sum(k-point weights) = ", sum(payload.basis.kweights))
for ik in eachindex(occupations_cpu)
    @printf("k-point=%d spin=%d occupied_bands=%d extra_bands=%d sum_occupations=%.8e\n",
            ik, payload.basis.kpoints[ik].spin, length(occupied_masks[ik]),
            length(extra_masks[ik]), sum(occupations_cpu[ik]))
end
for (label, density) in pairs(reconstructions)
    @printf("%-38s norm=%.8e relative_error_vs_reference=%.8e\n",
            string(label), norm(density), relative_error(density, static_reference))
    for spin in axes(density, 4)
        @printf("  spin=%d norm=%.8e reference_norm=%.8e\n",
                spin, norm(@view density[:, :, :, spin]),
                norm(@view static_reference[:, :, :, spin]))
    end
end
@printf("relative_error_DFTK_reconstruction_vs_reference = %.8e\n",
        relative_error(dftk_reconstructed, static_reference))
@printf("relative_error_our_plus_DFTK_occupation_vs_reference = %.8e\n",
        relative_error(our_plus_with_dftk_occupation, static_reference))
@printf("relative_error_our_plus_zero_occupation_vs_reference = %.8e\n",
        relative_error(our_plus_with_zero_occupation, static_reference))
@printf("relative_error_manual_vs_reference = %.8e\n",
        relative_error(manual_dynamic, static_reference))
@printf("relative_error_manual_vs_DFTK_reconstruction = %.8e\n",
        relative_error(manual_dynamic, dftk_reconstructed))
@printf("relative_error_B_vs_A = %.8e\n",
        relative_error(our_plus_with_dftk_occupation, dftk_reconstructed))
@printf("relative_error_C_vs_B = %.8e\n",
        relative_error(our_plus_with_zero_occupation, our_plus_with_dftk_occupation))
@printf("relative_error_D_vs_A = %.8e\n",
        relative_error(manual_dynamic, dftk_reconstructed))
@printf("reference delta_psi norm = %.8e\n", component_norm(reference_result.δψ))
@printf("oracle plus delta_psi norm = %.8e\n",
        component_norm(static_limit.delta_psi_plus))
@printf("delta_psi difference norm = %.8e\n",
        component_norm(missing_orbital_variation))
@printf("delta_psi relative difference = %.8e\n",
        component_norm(missing_orbital_variation) /
        max(component_norm(reference_result.δψ), eps(Float64)))
@printf("reference delta_occupation norm = %.8e\n",
        component_norm(reference_result.δoccupation))
for ik in eachindex(reference_result.δoccupation)
    @printf("  k-point=%d delta_occupation norm=%.8e\n",
            ik, component_norm(reference_result.δoccupation[ik]))
end
@printf("delta_fermi_level = %s (norm=%s)\n",
        string(reference_result.δεF), string(component_norm(reference_result.δεF)))
@printf("DFTK orbital-only density norm = %.8e\n", norm(dftk_orbital_only))
@printf("DFTK occupation-only density norm = %.8e\n", norm(dftk_occupation_only))
@printf("missing orbital density component norm = %.8e\n", norm(missing_orbital_density))
println("DFTK returns no separate explicit occupied-state field; that term is included in δψ.")
println("The missing-orbital density above is reconstructed from δψ_DFTK - δψ_oracle.")

static_orbital_relative_error = relative_orbital_error(
    static_orbital_reference.delta_psi, reference_result.δψ)
println("=== Reconstructed DFTK static orbital-response decomposition ===")
for ik in eachindex(reference_result.δψ)
    kpt = payload.basis.kpoints[ik]
    dftk_δψ = reference_result.δψ[ik]
    ref_δψ = static_orbital_reference.delta_psi[ik]
    metrics = static_orbital_reference.per_kpoint[ik]
    @printf("k=%d spin=%d ||δψ_DFTK||=%.8e ||δψ_ref||=%.8e rel_diff=%.8e\n",
            ik, kpt.spin, norm(dftk_δψ), norm(ref_δψ),
            relative_error(ref_δψ, dftk_δψ))
    @printf("  occupied=%d extra=%d explicit=%.8e extra_contribution=%.8e complement=%.8e residual=%.8e cg_residual=%.8e\n",
            metrics.n_occupied, metrics.n_extra, metrics.explicit_norm,
            metrics.extra_norm, metrics.complement_norm,
            metrics.residual, metrics.cg_residual)
end
@printf("complete static delta_psi relative difference = %.8e\n",
        static_orbital_relative_error)
@assert static_orbital_relative_error < 1e-6

static_relative_error = relative_error(manual_dynamic, static_reference)

println("=== DFTK dynamic Sternheimer oracle (validation only) ===")
println("static-limit z = ", z_static)
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
