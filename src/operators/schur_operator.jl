"""
    apply_schur(M, z, v)

S(z) v = (A - zI) v - Σ(z) v, with Σ(z) = B (D - zI)⁻¹ B†.
"""
apply_schur(M::SchurModel, z::Number, v) =
    M.A * v - z * v - apply_self_energy(M, z, v)

"""
    schur_matrix(M, z)

Dense S(z) = A - zI - B (D - zI)⁻¹ B†.
"""
schur_matrix(M::SchurModel, z::Number) = Matrix(M.A) - z * I - self_energy(M, z)

"""
    solve_schur(M, z, b)

Direct solve of S(z) x = b (exact Schur system, no approximation).
"""
solve_schur(M::SchurModel, z::Number, b) = schur_matrix(M, z) \ b
