"""
DFTKBridge: abstraction layer between DFTK-style Hamiltonian data and the
[`SchurModel`](@ref) used by this package.

Only dense reference implementations are provided; DFTK.jl is *not* a
dependency. Convention: `z = ω + iη`, shifted operators are `D - zI`.
A future path will support DFTK v0.7+ `HamiltonianBlock`s (via
[`linear_operator_adapter`](@ref)) without changing the Schur-model interface.
"""

"""
    DFTKOccVirtPartition(n_occ, n_virt; kpoint=nothing, metadata=nothing)

Sizes of the occupied (retained) and virtual (eliminated) spaces, with
optional/future metadata such as the k-point index (types are `Any`).
"""
struct DFTKOccVirtPartition
    n_occ::Int
    n_virt::Int
    kpoint::Any
    metadata::Any
end
DFTKOccVirtPartition(n_occ::Integer, n_virt::Integer; kpoint = nothing, metadata = nothing) =
    DFTKOccVirtPartition(Int(n_occ), Int(n_virt), kpoint, metadata)

"""
    build_schur_model(A, B, D)

Thin wrapper returning `SchurModel(A, D, B)` (note the argument order).
Future DFTK integration will build `A`, `B`, `D` from a DFTK SCF result.
"""
build_schur_model(A::AbstractMatrix, B::AbstractMatrix, D::AbstractMatrix) = SchurModel(A, D, B)

"""
    build_schur_model_from_blocks(H, occ_mask, virt_mask)

Dense reference: for a Hermitian block matrix `H = [A B; B' D]`, slice the
retained (occupied) block `A = H[occ, occ]`, coupling `B = H[occ, virt]` and
eliminated (virtual) block `D = H[virt, virt]` using Boolean masks, and return
`SchurModel(A, D, B)`. The masks must have length `size(H, 1)`, be disjoint and
non-empty, and `H` must be square. The shifted operator is `D - zI`, `z = ω + iη`.
"""
function build_schur_model_from_blocks(H::Matrix, occ_mask, virt_mask)
    n = size(H, 1)
    size(H, 2) == n || throw(DimensionMismatch("H must be square"))
    length(occ_mask) == n && length(virt_mask) == n ||
        throw(DimensionMismatch("masks must have length size(H, 1)"))
    occ = collect(Bool, occ_mask)
    virt = collect(Bool, virt_mask)
    any(occ .& virt) && throw(ArgumentError("occ_mask and virt_mask must be disjoint"))
    (any(occ) && any(virt)) || throw(ArgumentError("masks must each select at least one index"))
    return SchurModel(H[occ, occ], H[virt, virt], H[occ, virt])
end

"""
    linear_operator_adapter(M)

Wrap a matrix as a `LinearMaps.LinearMap` supporting `M*x` and `M'*x`. The
future DFTK path will build a LinearMap around `ham.blocks[ik] * ψ`.
"""
linear_operator_adapter(M::AbstractMatrix) = LinearMap(M)
