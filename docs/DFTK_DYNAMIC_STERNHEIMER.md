# DFTK 0.8.1 static Sternheimer reference and dynamic oracle

This note records the static reference inspected in
[`DFTK.jl` v0.8.1 `src/response/chi0.jl`](https://github.com/JuliaMolSim/DFTK.jl/blob/v0.8.1/src/response/chi0.jl)
and `src/densities.jl`. The installed source was not available in the
development environment; these equations and interfaces were checked against
the upstream `v0.8.1` tag. The implementation uses no DFTK dynamic-response
API.

## Static equation and representations

In `sternheimer_solver` (`chi0.jl:109-218`), for an occupied state with
eigenvalue `ε`, DFTK solves

```math
Q(H_k-\epsilon)Q\,\delta\psi=-Q\,\mathrm{rhs},\qquad
P=\psi_k\psi_k^\dagger,\quad Q=I-P.
```

In `compute_δψ!` (`chi0.jl:361-430`), `rhs` is the perturbation applied to the
occupied wavefunctions, `δHψ = δH ψ_{k-q}`. At `q=0`, the source and target
k-point coincide. DFTK's solver also accepts unconverged extra bands and uses
a Schur-complement split of their span from the uncomputed complement; the
equation above is the underlying projected problem. It projects the RHS with
`Q`, applies projectors during the solve, and returns a correction orthogonal
to the converged occupied subspace.

The objects are represented as follows:

- `H_k` is `ham.blocks[ik]`, a DFTK Hamiltonian acting on plane-wave
  coefficient vectors at one k-point.
- `epsilon_nk` is a vector of band eigenvalues for that k-point.
- `psi[ik]` is a plane-wave-coefficient matrix (basis functions by bands).
- `rhs[ik] = δHψ[ik]` and `δψ[ik]` have the same matrix layout; columns are
  band right-hand sides. DFTK's static solver handles multiple columns.

For `q=0`, `compute_δρ` (`densities.jl:60-111`) reconstructs the static density
change in real space from inverse-FFT wavefunctions:

```math
\delta\rho(\mathbf r)=
\sum_{k n} w_k\,\mathrm{occ}_{nk}\,
2\operatorname{Re}\!\left[\psi_{nk}^*(\mathbf r)\delta\psi_{nk}(\mathbf r)\right]
+\sum_{k n}w_k\,\delta\mathrm{occ}_{nk}|\psi_{nk}(\mathbf r)|^2.
```

The implementation applies the inverse-FFT normalization squared, reduces over
k-point MPI ranks, and symmetrizes the resulting density. `apply_χ0` also
symmetrizes and normalizes its input potential, forms `δHψ` with
`apply_potential_blochwave`, then calls `compute_δρ` (`chi0.jl:564-623`).

## Static orbital-response decomposition

The static orbital response in `compute_δψ!` is not just the projected
Sternheimer solution. For occupied input bands it is the sum

```math
\delta\psi_k =
\underbrace{\psi_{\rm occ}
  \left[\alpha \odot (\psi_{\rm occ}^\dagger \delta H\psi_{\rm occ})\right]}
_{\text{explicit occupied-state contribution}}
+
\underbrace{\psi_{\rm extra}\alpha_{\rm extra}+\delta\psi_R}
_{\text{extra-band and complement Sternheimer contribution}}.
```

The explicit matrix is formed in `chi0.jl:402-416`. For each occupied pair
`m != n`, DFTK computes
`ratio = filled_occ * occupation_divided_difference(...)` and
`α[m,n] = ratio * f_n / (f_n^2 + f_m^2)`. For q=0, the diagonal `m == n` is
skipped; its response is represented by `δoccupation`, not by this explicit
orbital term. The expression is added to the output before the Sternheimer
solve.

The second contribution is returned by `sternheimer_solver`
(`chi0.jl:115-218`). DFTK splits the orthogonal complement of the converged
occupied states into the computed extra-state span and the remaining
complement. Its Schur-complement solve returns
`ψ_extra * α_extra + δψ_R`; projecting this result onto `ψ_extra` isolates the
extra-band part, and subtracting that projection isolates the complement part.
`apply_χ0_4P` selects occupied and extra bands from the occupations and calls
`compute_δψ!` with these terms (`chi0.jl:452-510`). The static reference helper
uses the same occupation divided-difference formula, DFTK band tolerances,
extra-state Rayleigh-Ritz energies, and `sternheimer_solver`; the example
reports each component and compares the complete orbital response against
`DFTK.apply_χ0(...).δψ` before proceeding to dynamic frequencies.

## Reuse boundary for the reference oracle

The oracle reuses the SCF result's `ham.blocks`, wavefunctions, eigenvalues,
occupations, DFTK's `apply_potential_blochwave`, inverse FFT, and symmetry
operations. It solves the two shifted projected linear systems directly as
dense reference solves. This avoids copying DFTK's CG/preconditioner and
extra-band Schur-complement implementation, while keeping the same Hamiltonian,
occupied projectors, perturbation, and density convention. The static result
from `DFTK.apply_χ0` remains the independent regression reference.

For `z = ω + iη`, the two Sternheimer branches used to reconstruct the
frequency-dependent density are

```math
Q(H_k-\epsilon_{nk}-z)Q\,\delta\psi^+_{nk}=-Q\,\delta H\psi_{nk},
```

```math
Q(H_k-\epsilon_{nk}+\bar z)Q\,\delta\psi^-_{nk}=-Q\,\delta H\psi_{nk},
```

and the orbital density is assembled as
`occ * (conj(ψ) * δψ⁺ + ψ * conj(δψ⁻))`. As `z → 0`, both branches reduce to
the static Sternheimer correction and the density reduces to the static
`2Re` expression. This is a response-oracle validation only, not a claim of a
physical dielectric spectrum.
