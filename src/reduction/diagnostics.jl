"""
    rational_compression_diagnostics(M, ωgrid, η, b, shifts_list; kind = :rational, rtol = 1e-10)

For each shift set `σ` in `shifts_list` (for `kind = :polynomial` the entries
are integers = number of polynomial steps), build the basis from the starting
block `M.B'`, reduce, and evaluate on `z = ω + iη`, `ω ∈ ωgrid`:

* `Σerr_max/mean` : ‖Σ(z)-Σ_k(z)‖_F/‖Σ(z)‖_F  (ROM approximation error)
* `xerr_max`      : ‖x_exact-x_ROM‖/‖x_exact‖, x_exact = `solve_schur`, x_ROM = `solve_reduced_schur`
* `lin_err_max`   : ‖x_aux-x_ROM‖/‖x_ROM‖ with x_aux = `solve_full_reduced_upper`
                    (algebraic linearization identity error, NOT approximation error)

Returns a vector of NamedTuples `(kind, η, nshifts, k, Σerr_max, Σerr_mean, xerr_max, lin_err_max)`.
"""
function rational_compression_diagnostics(M::SchurModel, ωgrid, η::Real, b, shifts_list;
                                          kind::Symbol = :rational, rtol::Real = 1e-10)
    rows = NamedTuple[]
    for sh in shifts_list
        Q = kind === :rational ? rational_krylov_basis(M.D, M.B', sh; rtol = rtol) :
                                 polynomial_krylov_basis(M.D, M.B', sh; rtol = rtol)
        RM = ReducedSchurModel(M, Q)
        es = Float64[]; ex = Float64[]; el = Float64[]
        for ω in ωgrid
            z = ω + im * η
            Σ = self_energy(M, z)
            push!(es, norm(Σ - reduced_self_energy(RM, z)) / norm(Σ))
            x = solve_schur(M, z, b)
            xr = solve_reduced_schur(RM, z, b)
            push!(ex, norm(x - xr) / norm(x))
            push!(el, norm(solve_full_reduced_upper(RM, z, b) - xr) / norm(xr))
        end
        push!(rows, (kind = kind, η = η, nshifts = sh isa Integer ? sh : length(sh),
                     k = size(Q, 2), Σerr_max = maximum(es), Σerr_mean = sum(es) / length(es),
                     xerr_max = maximum(ex), lin_err_max = maximum(el)))
    end
    return rows
end

"""
    print_diagnostics(rows; io = stdout)

Print a table of rows returned by [`rational_compression_diagnostics`](@ref).
"""
function print_diagnostics(rows; io::IO = stdout)
    println(io, "kind        eta      s    k    max Σerr    mean Σerr   max xerr    max lin(aux-ROM)")
    for r in rows
        println(io, rpad(string(r.kind), 11), " ", rpad(r.η, 8), " ", rpad(r.nshifts, 4), " ", rpad(r.k, 4), " ",
                join((rpad(string(round(v; sigdigits = 3)), 11) for v in (r.Σerr_max, r.Σerr_mean, r.xerr_max, r.lin_err_max)), " "))
    end
end
