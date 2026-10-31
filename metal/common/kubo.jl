
module MetalCommon
using LinearAlgebra, Random

struct MetalModel
    schur_model
    mu::Float64
    sigma::Float64
    kpoints::Int
end

function synthetic_metal_model(; m=500, n=50, seed=20241031, metal=:Al)
    rng = MersenneTwister(seed)
    nb = m - (metal==:Fe2MnAl ? 6 : 4)
    bulk = [range(-8.0, -1.5; length=nb÷2); range(1.5, 8.0; length=nb-nb÷2)]
    if metal==:Fe
        λ = sort([bulk; -0.8; -0.2; 0.4; 0.9])
    elseif metal==:Fe2MnAl
        # Heusler: 密なd共鳴、擬ギャップ近傍に6極
        λ = sort([bulk; -1.2; -0.6; -0.15; 0.15; 0.6; 1.1])
    else # Al, Al40
        λ = sort([bulk; -0.6; -0.1; 0.35; 0.8])
    end
    U = Matrix(qr(randn(rng, ComplexF64, m, m)).Q)
    D = U * Diagonal(λ) * U'
    D = (D + D')/2
    A = (X=randn(rng, ComplexF64, n, n); (X+X')/2)
    if metal==:Fe2MnAl
        w = [abs(x)<1.3 ? 0.7 : 3/sqrt(m) for x in λ]
    else
        w = [abs(x)<1.2 ? 0.6 : 3/sqrt(m) for x in λ]
    end
    B = randn(rng, ComplexF64, n, m) * Diagonal(w) * U'
    M = SchurModel(A,D,B)
    return MetalModel(M, 0.0, 0.1, 1)
end

end
