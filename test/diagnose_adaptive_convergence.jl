using LinearAlgebra, Printf, Random, ShiftSchur

function synthetic_rational_model(; m=150, n=3, seed=20240601)
    rng = MersenneTwister(seed)
    nb = m - 4
    bulk = [range(-3.0, -1.3; length=nb ÷ 2);
            range(1.3, 3.0; length=nb - nb ÷ 2)]
    λ = sort([bulk; -0.6; -0.1; 0.35; 0.8])
    U = Matrix(qr(randn(rng, ComplexF64, m, m)).Q)
    D = U * Diagonal(λ) * U'
    D = (D + D') / 2
    A0 = randn(rng, ComplexF64, n, n)
    A = (A0 + A0') / 2
    w = [abs(x) < 1.1 ? 0.5 : 3 / sqrt(m) for x in λ]
    B = randn(rng, ComplexF64, n, m) * Diagonal(w) * U'
    b = randn(rng, ComplexF64, n)
    return SchurModel(A, D, B), b
end

function true_max_error(M, shifts, ωgrid, η, b)
    Q = rational_krylov_basis(M.D, M.B', shifts)
    RM = ReducedSchurModel(M, Q)
    return maximum(ωgrid) do ω
        z = ω + im * η
        exact = apply_self_energy(M, z, b)
        approx = apply_reduced_self_energy(RM, z, b)
        denom = norm(exact)
        denom == 0 ? (norm(exact - approx) == 0 ? 0.0 : Inf) :
                     norm(exact - approx) / denom
    end
end

M, b = synthetic_rational_model()
ωgrid = range(-1.0, 1.0; length=41)
η = 0.1
init_shifts = ComplexF64[0.0 + im * η]

for max_poles in (16, 20)
    println("\n=== Adaptive convergence diagnostic: max_poles=$max_poles, tol=1e-4 ===")
    _, history = greedy_rational_shifts_aposteriori(
        M, ωgrid, η, b; s_max=max_poles - length(init_shifts),
        init_shifts=init_shifts, tol=1e-4)

    println("npoles | basis_dim | max_estimator | max_true_error | max_omega | selected_omega")
    for step in history
        true_error = true_max_error(M, step.shifts, ωgrid, η, b)
        selected = step.selected_omega === nothing ? "none" : string(step.selected_omega)
        @printf("%6d | %9d | %.8e | %.8e | %9.5f | %s\n",
                step.n_poles, step.k, step.max_err, true_error,
                step.max_omega, selected)
    end
end
