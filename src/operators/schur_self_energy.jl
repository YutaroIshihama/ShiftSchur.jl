"""
    SchurModel(A, D, B)

Hermitian `A` (n×n, retained space), Hermitian `D` (m×m, eliminated space)
and coupling `B` (n×m). Convention: `z = ω + iη`, shifts enter as `X - zI`.
"""
struct SchurModel{TA,TD,TB}
    A::TA
    D::TD
    B::TB
end

"""
    apply_self_energy(M, z, v)

Σ(z) v = B (D - zI)⁻¹ B† v  (matrix-compatible; one D-solve per call).
"""
apply_self_energy(M::SchurModel, z::Number, v) =
    M.B * ((M.D - z * I) \ (M.B' * v))

"""
    self_energy(M, z)

Dense Σ(z) = B (D - zI)⁻¹ B†.
"""
self_energy(M::SchurModel, z::Number) = M.B * ((M.D - z * I) \ Matrix(M.B'))
