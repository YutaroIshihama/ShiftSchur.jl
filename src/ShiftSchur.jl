module ShiftSchur

using LinearAlgebra

export SchurModel, self_energy, apply_self_energy, apply_schur, schur_matrix,
       solve_schur, auxiliary_matrix, solve_full_upper
export rational_krylov_basis, polynomial_krylov_basis, ReducedSchurModel,
       reduced_self_energy, apply_reduced_self_energy, reduced_schur_matrix,
       solve_reduced_schur, reduced_auxiliary_matrix, solve_full_reduced_upper,
       rational_compression_diagnostics, print_diagnostics

include("operators/schur_self_energy.jl")
include("operators/schur_operator.jl")
include("operators/auxiliary_operator.jl")
include("reduction/rational_krylov.jl")
include("reduction/projection.jl")
include("reduction/diagnostics.jl")

end
