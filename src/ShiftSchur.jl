module ShiftSchur

using LinearAlgebra

export SchurModel, self_energy, apply_self_energy, apply_schur, schur_matrix,
       solve_schur, auxiliary_matrix, solve_full_upper

include("operators/schur_self_energy.jl")
include("operators/schur_operator.jl")
include("operators/auxiliary_operator.jl")

end
