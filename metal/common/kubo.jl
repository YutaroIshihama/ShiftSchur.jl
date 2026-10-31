
module MetalCommon
using LinearAlgebra, Random
using ..ShiftSchur: SchurModel

struct MetalModel
    schur_model
    mu::Float64
    sigma::Float64
    kpoints::Int
end

function synthetic_metal_model(; m=500, n=50, seed=20241031, metal=:Al)
    m > 0 || throw(ArgumentError("m must be positive"))
    n > 0 || throw(ArgumentError("n must be positive"))
    metal in (:Al, :Fe2MnAl) ||
        throw(ArgumentError("metal must be :Al or :Fe2MnAl"))

    rng = MersenneTwister(seed)
    resonances = metal === :Fe2MnAl ?
        [-1.2, -0.6, -0.15, 0.15, 0.6, 1.1] :
        [-0.6, -0.1, 0.35, 0.8]
    bulk_size = m - length(resonances)
    bulk_size >= 2 ||
        throw(ArgumentError("m must leave at least two bulk states"))
    lower, upper = metal === :Fe2MnAl ? (-8.0, 8.0) : (-3.0, 3.0)
    gap_edge = metal === :Fe2MnAl ? 1.5 : 1.3
    lower_bulk = range(lower, -gap_edge; length=bulk_size ÷ 2)
    upper_bulk = range(gap_edge, upper; length=bulk_size - bulk_size ÷ 2)
    λ = sort([collect(lower_bulk); resonances; collect(upper_bulk)])
    U = Matrix(qr(randn(rng, ComplexF64, m, m)).Q)
    D = U * Diagonal(λ) * U'
    D = (D + D')/2
    A = (X=randn(rng, ComplexF64, n, n); (X+X')/2)
    w = [x in resonances ? 0.7 : 3/sqrt(m) for x in λ]
    B = randn(rng, ComplexF64, n, m) * Diagonal(w) * U'
    M = SchurModel(A,D,B)
    return MetalModel(M, 0.0, 0.1, 1)
end

end
