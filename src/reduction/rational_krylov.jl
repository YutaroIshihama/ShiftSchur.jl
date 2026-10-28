# Orthonormal basis of range(V) via SVD, discarding directions with
# singular value <= rtol * ||V||_F. Returns an m×r matrix (r may be 0).
function _orth_range(V::AbstractMatrix, rtol::Real)
    nrm = norm(V)
    (nrm == 0 || size(V, 2) == 0) && return zeros(eltype(V), size(V, 1), 0)
    F = svd(V)
    r = count(>(rtol * nrm), F.S)
    return F.U[:, 1:r]
end

# Orthonormal basis of range(V) ⟂ range(Q): two block Gram-Schmidt passes,
# rank-revealing SVD, then one more cleanup pass.
function _orth_against(Q::AbstractMatrix, V::AbstractMatrix, rtol::Real)
    nrm = norm(V)
    nrm == 0 && return zeros(eltype(V), size(V, 1), 0)
    R = V
    for _ in 1:2
        R = R - Q * (Q' * R)
    end
    F = svd(R)
    r = count(>(rtol * nrm), F.S)
    r == 0 && return zeros(eltype(V), size(V, 1), 0)
    U = F.U[:, 1:r]
    U = U - Q * (Q' * U)
    return Matrix(qr(U).Q)[:, 1:r]
end

function _krylov_basis(apply, W::AbstractMatrix, nsteps::Integer, rtol::Real)
    T = promote_type(eltype(W), ComplexF64)
    Wc = Matrix{T}(W)
    Vk = _orth_range(Wc, rtol)
    Q = Vk
    for j in 1:nsteps
        size(Vk, 2) == 0 && break
        # continue from the full (undeflated) orthonormal block, so the space
        # is exactly the stated Krylov space
        Vk = _orth_range(apply(j, Vk), rtol)
        Q = hcat(Q, _orth_against(Q, Vk, rtol))
    end
    return Q
end

"""
    rational_krylov_basis(D, W, shifts; rtol = 1e-10)

Orthonormal basis `Q` (m×k, `Q†Q ≈ I`) of the block rational Krylov space

    K = span{ W, (D-σ₁I)⁻¹W, (D-σ₂I)⁻¹(D-σ₁I)⁻¹W, …, ∏_{j=1}^{s}(D-σⱼI)⁻¹ W }

for Hermitian `D` (m×m), starting block `W` (m×p, e.g. `W = B†`) and the
explicitly supplied `shifts = [σ₁,…,σₛ]` (may be complex, must avoid the
spectrum of `D`). Each block is the previous block (not only its new part)
multiplied by `(D-σⱼI)⁻¹`. Equivalently
`K = { r(D) w : w ∈ range(W), r = p/∏(z-σⱼ), deg p ≤ s }` — a rational Krylov
space with prescribed poles σⱼ (single sequential pole sequence, no adaptivity).

Rank-revealing: each block is orthonormalized by SVD, directions with singular
value ≤ `rtol`·‖block‖_F are dropped, and orthogonalization against the
existing basis is done with repeated Gram-Schmidt. Hence `k ≤ p(s+1)`; a zero
or rank-deficient `W` yields fewer (possibly zero) columns. Dense proof of
concept (one LU of `D-σⱼI` per shift).
"""
function rational_krylov_basis(D::AbstractMatrix, W::AbstractMatrix, shifts; rtol::Real = 1e-10)
    σ = collect(shifts)
    Dd = Matrix(D)
    facts = [lu(Dd - s * I) for s in σ]
    return _krylov_basis((j, V) -> facts[j] \ V, W, length(σ), rtol)
end

"""
    polynomial_krylov_basis(D, W, nsteps; rtol = 1e-10)

Orthonormal basis of the block polynomial Krylov space
`span{W, DW, D²W, …, D^nsteps W}`, built with the same rank-revealing
orthogonalization as [`rational_krylov_basis`](@ref). Used as a comparison baseline.
"""
function polynomial_krylov_basis(D::AbstractMatrix, W::AbstractMatrix, nsteps::Integer; rtol::Real = 1e-10)
    Dd = Matrix(D)
    return _krylov_basis((j, V) -> Dd * V, W, nsteps, rtol)
end
