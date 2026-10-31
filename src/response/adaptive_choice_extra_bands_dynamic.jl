"""
Dynamic counterpart of the static adaptive extra-band choice.

The static estimate scales like `N_extra ~ σ/gap`. Dynamically, the required
space depends on the sampled frequency: `N_extra(ω)` grows when
`|ε_c - ε_v - ω| < η`, so the greedy shifts are selected from the frequency
grid rather than from a single static gap.
"""
struct AdaptiveExtraBandsResult
    n_extra::Int
    k_dim::Int
    shifts::Vector{ComplexF64}
    max_err::Float64
    history::Vector{NamedTuple}
end

"""
    adaptive_extra_bands_dynamic(M, ωgrid, η, b; tol=1e-8, s_max=16)

Greedily choose shifts until the maximum residual estimate over `ωgrid`
falls below `tol` or `s_max` shifts have been added.
"""
function adaptive_extra_bands_dynamic(M::SchurModel, ωgrid, η::Real, b;
                                      tol::Real=1e-8, s_max::Integer=16,
                                      rtol::Real=1e-10)
    shifts, history = greedy_rational_shifts_aposteriori(
        M, ωgrid, η, b; tol=tol, s_max=s_max, rtol=rtol)
    Q = rational_krylov_basis(M.D, M.B', shifts; rtol=rtol)
    RM = ReducedSchurModel(M, Q)
    DQ = ErrorEstimators.compute_DQ(M.D, Q)
    max_err = maximum(ErrorEstimators.residual_norms_grid(
        RM, DQ, ωgrid, η, b, M.B))

    return AdaptiveExtraBandsResult(size(Q, 2), size(Q, 2), shifts,
                                    max_err, history)
end

"""
    adaptive_choice_vs_default_comparison(M, ωgrid, η, b; tol=1e-8, s_max=16)

Compare the residual-estimated maximum error and reduced basis dimension for
four fixed default shifts and the dynamically selected shifts.
"""
function adaptive_choice_vs_default_comparison(M::SchurModel, ωgrid, η::Real, b;
                                               tol::Real=1e-8,
                                               s_max::Integer=16,
                                               rtol::Real=1e-10)
    default_shifts = ComplexF64.(collect(range(-1, 1; length=4) .+ im * η))
    Qdefault = rational_krylov_basis(M.D, M.B', default_shifts; rtol=rtol)
    RMdefault = ReducedSchurModel(M, Qdefault)
    DQdefault = ErrorEstimators.compute_DQ(M.D, Qdefault)
    default_err = maximum(ErrorEstimators.residual_norms_grid(
        RMdefault, DQdefault, ωgrid, η, b, M.B))

    adaptive = adaptive_extra_bands_dynamic(
        M, ωgrid, η, b; tol=tol, s_max=s_max, rtol=rtol)
    return (default=(k=size(Qdefault, 2), max_err=default_err),
            adaptive=(k=adaptive.k_dim, max_err=adaptive.max_err,
                      result=adaptive))
end
