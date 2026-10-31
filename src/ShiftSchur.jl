module ShiftSchur

using LinearAlgebra
using LinearMaps

export SchurModel, self_energy, apply_self_energy, apply_schur, schur_matrix,
       solve_schur, auxiliary_matrix, solve_full_upper
export rational_krylov_basis, polynomial_krylov_basis, ReducedSchurModel,
       reduced_self_energy, apply_reduced_self_energy, reduced_schur_matrix,
       solve_reduced_schur, reduced_auxiliary_matrix, solve_full_reduced_upper,
       rational_compression_diagnostics, print_diagnostics, greedy_rational_shifts,
       greedy_rational_shifts_aposteriori, adaptive_extra_bands_dynamic,
       adaptive_choice_vs_default_comparison
export build_schur_model, build_schur_model_from_blocks, linear_operator_adapter,
       chi0_reduced, dielectric_matrix_reduced, macroscopic_epsilon, DFTKOccVirtPartition
export dynamic_sternheimer_response

include("operators/schur_self_energy.jl")
include("operators/schur_operator.jl")
include("operators/auxiliary_operator.jl")
include("reduction/rational_krylov.jl")
include("reduction/projection.jl")
include("reduction/error_estimators.jl")
include("reduction/adaptive_poles.jl")
include("response/adaptive_choice_extra_bands_dynamic.jl")
include("reduction/diagnostics.jl")
include("dftk/bridge.jl")
include("dftk/dynamic_sternheimer.jl")
using .ShiftSchurDFTKDynamicSternheimer: dynamic_sternheimer_response
include("dielectric/dielectric.jl")

end
