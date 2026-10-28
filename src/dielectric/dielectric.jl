"""
Toy reduced-model dielectric reference (NOT a production DFTK dielectric
calculation). Uses `χ0_k(z) = -reduced_self_energy(RM, z)` and
`ε_k(z) = I - Vc*χ0_k(z)`, with `z = ω + iη` and the `D - zI` convention.
The sign should be aligned with the eventual DFTK convention.
"""

"""
    chi0_reduced(RM, z)

Toy reduced response `χ0(z) = -reduced_self_energy(RM, z)`.
"""
chi0_reduced(RM::ReducedSchurModel, z::Number) = -reduced_self_energy(RM, z)

"""
    dielectric_matrix_reduced(RM, z, Vc)
    dielectric_matrix_reduced(RM, z)

`ε(z) = I - Vc*χ0(z)` for an `n×n` Coulomb matrix `Vc`; without `Vc` the
identity is used.
"""
function dielectric_matrix_reduced(RM::ReducedSchurModel, z::Number, Vc::AbstractMatrix)
    χ = chi0_reduced(RM, z)
    n = size(χ, 1)
    size(Vc) == (n, n) || throw(DimensionMismatch("Vc must be $n×$n"))
    return I - Vc * χ
end
function dielectric_matrix_reduced(RM::ReducedSchurModel, z::Number)
    n = size(RM.Bred, 1)
    return dielectric_matrix_reduced(RM, z, Matrix{eltype(RM.Bred)}(I, n, n))
end

"""
    macroscopic_epsilon(ε_mat)

Toy head average `1 / inv(ε_mat)[1,1]`; reference for a future DFTK
macroscopic-epsilon calculation.
"""
macroscopic_epsilon(ε_mat::AbstractMatrix)::Number = 1 / inv(ε_mat)[1, 1]
