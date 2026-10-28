"""
    ReducedSchurModel(A, T, Bred, Q)
    ReducedSchurModel(M::SchurModel, Q)

Galerkin-projected model with `T = Q†DQ` (k×k Hermitian) and `Bred = BQ`
(n×k). Convention: `z = ω + iη`, shifts enter as `X - zI`. The exact
[`SchurModel`](@ref) is left untouched.
"""
struct ReducedSchurModel{TA,TT,TB,TQ}
    A::TA
    T::TT
    Bred::TB
    Q::TQ
end

function ReducedSchurModel(M::SchurModel, Q::AbstractMatrix)
    ReducedSchurModel(M.A, Q' * (M.D * Q), M.B * Q, Q)
end

"""
    reduced_self_energy(RM, z)

Dense Σ_k(z) = Bred (T - zI)⁻¹ Bred†.
"""
reduced_self_energy(RM::ReducedSchurModel, z::Number) =
    RM.Bred * ((Matrix(RM.T) - z * I) \ Matrix(RM.Bred'))

"""
    apply_reduced_self_energy(RM, z, v)

Σ_k(z) v = Bred (T - zI)⁻¹ Bred† v.
"""
apply_reduced_self_energy(RM::ReducedSchurModel, z::Number, v) =
    RM.Bred * ((Matrix(RM.T) - z * I) \ (RM.Bred' * v))

"""
    reduced_schur_matrix(RM, z)

Dense S_k(z) = A - zI - Σ_k(z).
"""
reduced_schur_matrix(RM::ReducedSchurModel, z::Number) =
    Matrix(RM.A) - z * I - reduced_self_energy(RM, z)

"""
    solve_reduced_schur(RM, z, b)

Direct solve of S_k(z) x = b (approximate Schur system from the reduced model).
"""
solve_reduced_schur(RM::ReducedSchurModel, z::Number, b) = reduced_schur_matrix(RM, z) \ b

"""
    reduced_auxiliary_matrix(RM)

H_aux,k = [A Bred; Bred† T]; (H_aux,k - zI) is the enlarged system whose
Schur complement is S_k(z).
"""
reduced_auxiliary_matrix(RM::ReducedSchurModel) =
    [Matrix(RM.A) Matrix(RM.Bred); Matrix(RM.Bred') Matrix(RM.T)]

"""
    solve_full_reduced_upper(RM, z, b)

Upper block x_aux of the solution of `(H_aux,k - zI)[x; y] = [b; 0]`.
Algebraically equal to `solve_reduced_schur(RM, z, b)` (linearization
identity, independent of how well the ROM approximates the exact model).
"""
function solve_full_reduced_upper(RM::ReducedSchurModel, z::Number, b)
    n = size(RM.A, 1)
    H = reduced_auxiliary_matrix(RM)
    rhs = [b; zeros(eltype(H), size(RM.T, 1), size(b)[2:end]...)]
    sol = (H - z * I) \ rhs
    sol[1:n, ntuple(_ -> :, ndims(b) - 1)...]
end
