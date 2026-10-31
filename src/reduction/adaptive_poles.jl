"""
    greedy_rational_shifts(M, ωgrid, η, b; s_max=12, rtol=1e-10, init_shifts=ComplexF64[im * η])

Greedily select rational Krylov shifts for the self-energy action on `b`,
using an a posteriori residual estimate without evaluating the exact
self-energy. The process stops when the maximum estimate is below `1e-8`
or after `s_max` shifts have been added.

Returns the selected shifts and a vector of per-iteration named tuples with
`max_err` and basis dimension `k`.
"""
function greedy_rational_shifts(M::SchurModel, ωgrid, η::Real, b;
                                s_max::Integer = 12, rtol::Real = 1e-10,
                                init_shifts = ComplexF64[im * η],
                                tol::Real = 1e-8)
    return greedy_rational_shifts_aposteriori(
        M, ωgrid, η, b; s_max = s_max, rtol = rtol,
        init_shifts = init_shifts, tol = tol)
end

"""
    greedy_rational_shifts_aposteriori(M, ωgrid, η, b; s_max=16, rtol=1e-10)

Select rational Krylov shifts using the residual estimator without evaluating
`self_energy(M, z) * b`. The cached `D*Q` is extended only for newly added
basis columns.
"""
function greedy_rational_shifts_aposteriori(M::SchurModel, ωgrid, η::Real, b;
                                            s_max::Integer = 16, rtol::Real = 1e-10,
                                            init_shifts = ComplexF64[im * η],
                                            estimator::Symbol = :residual,
                                            tol::Real = 1e-8)
    s_max >= 0 || throw(ArgumentError("s_max must be nonnegative"))
    tol >= 0 || throw(ArgumentError("tol must be nonnegative"))
    estimator === :residual ||
        throw(ArgumentError("only the :residual estimator is implemented"))
    frequencies = collect(ωgrid)
    isempty(frequencies) && throw(ArgumentError("ωgrid must not be empty"))

    shifts = ComplexF64.(collect(init_shifts))
    history = NamedTuple[]
    Q = rational_krylov_basis(M.D, M.B', shifts; rtol = rtol)
    RM = ReducedSchurModel(M, Q)
    DQ = ErrorEstimators.compute_DQ(M.D, Q)

    for added in 0:s_max
        errors = ErrorEstimators.residual_norms_grid(RM, DQ, frequencies, η, b, M.B)
        max_err, worst = findmax(errors)
        if max_err <= tol || added == s_max
            push!(history, (max_err = max_err, k = size(Q, 2),
                            est_type = estimator, selected_omega = nothing,
                            shifts = copy(shifts)))
            @info("adaptive pole iteration", added=added, max_err=max_err,
                  selected_omega=nothing, basis_dimension=size(Q, 2),
                  shifts=copy(shifts))
            break
        end

        candidate = nothing
        for idx in sortperm(errors; rev = true)
            pole = ComplexF64(frequencies[idx] + im * η)
            if all(abs(pole - σ) > 100eps(Float64) * max(1.0, abs(pole), abs(σ))
                   for σ in shifts)
                candidate = pole
                break
            end
        end
        if candidate === nothing
            push!(history, (max_err = max_err, k = size(Q, 2),
                            est_type = estimator, selected_omega = nothing,
                            shifts = copy(shifts)))
            @info("adaptive pole iteration", added=added, max_err=max_err,
                  selected_omega=nothing, basis_dimension=size(Q, 2),
                  shifts=copy(shifts))
            break
        end

        push!(shifts, candidate)
        selected_omega = real(candidate)
        push!(history, (max_err = max_err, k = size(Q, 2),
                        est_type = estimator, selected_omega = selected_omega,
                        shifts = copy(shifts)))
        @info("adaptive pole iteration", added=added, max_err=max_err,
              selected_omega=selected_omega, basis_dimension=size(Q, 2),
              shifts=copy(shifts))

        Qnew = rational_krylov_basis(M.D, M.B', shifts; rtol = rtol)
        DQ = ErrorEstimators.expand_DQ(M.D, Q, DQ, Qnew)
        Q = Qnew
        RM = ReducedSchurModel(M, Q)
    end
    return shifts, history
end
