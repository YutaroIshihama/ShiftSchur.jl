"""
    greedy_rational_shifts(M, ωgrid, η, b; s_max=12, rtol=1e-10, init_shifts=ComplexF64[im * η])

Greedily select rational Krylov shifts for the self-energy action on `b`.
Starting from `init_shifts`, repeatedly build a basis from `M.D` and `M.B'`,
then add the shift `ω + im*η` at the grid point with largest relative error
`norm((Σ(z) - Σₖ(z)) * b) / norm(Σ(z) * b)`, where `z = ω + im*η`,
`Σ(z) = self_energy(M, z)`, and `Σₖ(z) = reduced_self_energy(RM, z)`.
The process stops when the maximum error is below `1e-8` or after `s_max`
shifts have been added. Repeated candidate shifts are slightly perturbed.

Returns the selected shifts and a vector of per-iteration named tuples with
`max_err` and basis dimension `k`.
"""
function greedy_rational_shifts(M::SchurModel, ωgrid, η::Real, b;
                                s_max::Integer = 12, rtol::Real = 1e-10,
                                init_shifts = ComplexF64[im * η])
    s_max >= 0 || throw(ArgumentError("s_max must be nonnegative"))
    frequencies = collect(ωgrid)
    isempty(frequencies) && throw(ArgumentError("ωgrid must not be empty"))

    shifts = ComplexF64.(collect(init_shifts))
    history = NamedTuple[]
    for added in 0:s_max
        Q = rational_krylov_basis(M.D, M.B', shifts; rtol = rtol)
        RM = ReducedSchurModel(M, Q)
        errors = Float64[]
        for ω in frequencies
            z = ω + im * η
            exact = self_energy(M, z) * b
            approx = reduced_self_energy(RM, z) * b
            denominator = norm(exact)
            err = denominator == 0 ? (norm(exact - approx) == 0 ? 0.0 : Inf) :
                                     norm(exact - approx) / denominator
            push!(errors, err)
        end

        max_err, worst = findmax(errors)
        push!(history, (max_err = max_err, k = size(Q, 2)))
        (max_err < 1e-8 || added == s_max) && break

        candidate = ComplexF64(frequencies[worst] + im * η)
        while any(abs(candidate - σ) <= 100eps(Float64) * max(1.0, abs(candidate), abs(σ))
                  for σ in shifts)
            candidate += sqrt(eps(Float64)) * max(1.0, abs(candidate))
        end
        push!(shifts, candidate)
    end
    return shifts, history
end
