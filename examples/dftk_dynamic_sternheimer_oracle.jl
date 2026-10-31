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
deltaV = static_smoke.deltaV
reference_result = static_smoke.raw

get_field(ref, sym_unicode, sym_ascii) = hasproperty(ref, sym_unicode) ? getproperty(ref, sym_unicode) : (hasproperty(ref, sym_ascii) ? getproperty(ref, sym_ascii) : nothing)

static_reference = get_field(reference_result, :δρ, :delta_rho)
get_dpsi(ref) = get_field(ref, :δψ, :delta_psi)
get_docc(ref) = get_field(ref, :δoccupation, :delta_occupation)
get_dEF(ref) = get_field(ref, :δεF, :delta_epsilon_F)

z_static = 1e-8im
static_limit = dynamic_sternheimer_response(payload.raw, deltaV, z_static; dftk_dpsi_ref=get_dpsi(reference_result))
occupation_threshold = DFTK.default_occupation_threshold(eltype(deltaV))
occupations_cpu = [DFTK.to_cpu(occ) for occ in payload.raw.occupation]
occupied_masks, extra_masks = DFTK.occupied_empty_masks(
    occupations_cpu, occupation_threshold)
zero_delta_psi = zero.(get_dpsi(reference_result))
zero_delta_occupation = zero.(get_docc(reference_result))

# For proper comparison, reconstruct_density must include norm factor like DFTK does
norm_deltaV = hasproperty(static_limit, :norm_deltaV) ? static_limit.norm_deltaV : norm(DFTK.symmetrize_ρ(payload.basis, deltaV))
println("norm_deltaV used for scaling = $norm_deltaV")

reconstruct_density(dpsi, docc) = DFTK.compute_δρ(
    payload.basis, payload.psi, dpsi, payload.raw.occupation, docc;
    occupation_threshold=occupation_threshold) * norm_deltaV

dftk_reconstructed = reconstruct_density(get_dpsi(reference_result),
                                           get_docc(reference_result))
our_plus_with_dftk_occupation = reconstruct_density(
    static_limit.delta_psi_plus, get_docc(reference_result))
our_plus_with_zero_occupation = reconstruct_density(
    static_limit.delta_psi_plus, zero_delta_occupation)
dftk_orbital_only = reconstruct_density(get_dpsi(reference_result), zero_delta_occupation)
dftk_occupation_only = reconstruct_density(zero_delta_psi,
                                            get_docc(reference_result))
missing_orbital_variation = map(-, get_dpsi(reference_result),
                                static_limit.delta_psi_plus)
missing_orbital_density = reconstruct_density(missing_orbital_variation,
                                              zero_delta_occupation)
manual_dynamic = static_limit.delta_rho !== nothing ? static_limit.delta_rho : our_plus_with_dftk_occupation
# If static_limit.delta_rho is orbital-only, we need to add occupation part for full comparison
# For now, use our_plus_with_dftk_occupation as manual
if hasproperty(static_limit, :delta_rho) && static_limit.delta_rho !== nothing
    # static_limit.delta_rho is orbital-only scaled, need to add occupation
    # So manual = orbital + occupation
    manual_dynamic = our_plus_with_dftk_occupation
end

relative_error(a, b) = norm(a - b) / max(norm(b), eps(Float64))
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

println("=== DFTK.apply_chi0 result fields ===")
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
@printf("reference delta_psi norm = %.8e\n", component_norm(get_dpsi(reference_result)))
@printf("oracle plus delta_psi norm = %.8e\n",
        component_norm(static_limit.delta_psi_plus))
@printf("delta_psi difference norm = %.8e\n",
        component_norm(missing_orbital_variation))
@printf("delta_psi relative difference = %.8e\n",
        component_norm(missing_orbital_variation) /
        max(component_norm(get_dpsi(reference_result)), eps(Float64)))
@printf("reference delta_occupation norm = %.8e\n",
        component_norm(get_docc(reference_result)))
for ik in eachindex(get_docc(reference_result))
    @printf("  k-point=%d delta_occupation norm=%.8e\n",
            ik, component_norm(get_docc(reference_result)[ik]))
end
@printf("delta_fermi_level = %s (norm=%s)\n",
        string(get_dEF(reference_result)), string(component_norm(get_dEF(reference_result))))
@printf("DFTK orbital-only density norm = %.8e\n", norm(dftk_orbital_only))
@printf("DFTK occupation-only density norm = %.8e\n", norm(dftk_occupation_only))
@printf("missing orbital density component norm = %.8e\n", norm(missing_orbital_density))
println("DFTK returns no separate explicit occupied-state field; that term is included in delta_psi.")
println("The missing-orbital density above is reconstructed from delta_psi_DFTK - delta_psi_oracle.")

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
for (omega, eta) in ((0.00, 0.05), (0.15, 0.05), (0.30, 0.05))
    response = dynamic_sternheimer_response(payload.raw, deltaV, omega + im * eta)
    rho = response.delta_rho !== nothing ? response.delta_rho : reconstruct_density(response.delta_psi_plus, get_docc(reference_result))
    push!(rows, (omega=omega, eta=eta, response_norm=norm(rho),
                 orbital_residual=response.orbital_residual,
                 plus_residual=response.branch_residuals.plus,
                 minus_residual=response.branch_residuals.minus,
                 orthogonality_error=response.occupied_orthogonality_error))
    @printf("omega=%.2f eta=%.2f norm(delta_rho)=%.8e residual=%.8e orthogonality=%.8e\n",
            omega, eta, rows[end].response_norm, response.orbital_residual,
            response.occupied_orthogonality_error)
end

output = joinpath(@__DIR__, "..", "docs", "dftk_dynamic_sternheimer_oracle.csv")
open(output, "w") do io
    println(io, "omega,eta,response_norm,orbital_residual,plus_residual,minus_residual,orthogonality_error")
    rho0 = static_limit.delta_rho !== nothing ? static_limit.delta_rho : manual_dynamic
    # For CSV, use full density (with occupation) for static limit
    rho0_full = our_plus_with_dftk_occupation
    println(io, "0.0,$(imag(z_static)),$(norm(rho0_full))," *
               "$(static_limit.orbital_residual),$(static_limit.branch_residuals.plus)," *
               "$(static_limit.branch_residuals.minus),$(static_limit.occupied_orthogonality_error)")
    for row in rows
        println(io, "$(row.omega),$(row.eta),$(row.response_norm),$(row.orbital_residual)," *
                   "$(row.plus_residual),$(row.minus_residual),$(row.orthogonality_error)")
    end
end
println("Saved response norms and residuals to $output")
println("This oracle output is not a physical dielectric spectrum.")
