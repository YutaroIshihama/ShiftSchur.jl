# Theory and reference-code mapping (Phase 1)

Convention: z = ω + iη; S(z) = A − zI − B (D − zI)⁻¹ B†; the full system
[A−zI B; B† D−zI][x; y] = [b; 0] has upper block x solving S(z)x = b.

## Mapping from `sternheimer_solver_dynamic` (reference)
- z = branch·ω + iη.
- Retained space: R = I − P_occ − P_extra; A ↔ R (Hk − (ε_n + z)) R (`A(phi)`, `RAR`).
- D ↔ diagonal `d = eps_extra − eps_n − z` (extra orbitals, exact eigenpairs; `solveD`).
- B ↔ R (Hk − ε_n − z) ψ_extra = R·Hpsi_extra (`Aex`; R ψ_extra = 0 so the shift drops out).
- b ↔ −Q(rhs) projected: `bb = R(b − Aex D⁻¹ ψ_extra† b)`; x ↔ `dR`; dpsi = ψ_extra α + dR.

## Mismatches with the idealized formula
1. Reference `A` already contains the shift (ε_n + z), so A_ideal = R(Hk − ε_n)R and the
   formula A − zI matches only on range(R); D has ε_n subtracted (D_ideal = ε_extra − ε_n).
2. Hpsi_extra with exact extra eigenpairs: R Hk ψ_extra = 0 mathematically, so B ≈ 0
   (up to eigensolver tolerance) and the Schur correction is not the full coupling; with
   inexact (LOBPCG) extra bands B is a small residual coupling. Σ(z) is then rational with
   poles at ε_extra − ε_n.
3. The Schur correction uses `Aex' * r` (Aex† applied to R-projected r) — consistent with B†,
   but `Aex'` conjugates only the matrix, and with complex z the term (ε_n+z)ψ_extra is
   removed by R; the implementation is Hermitian-structured only if z is treated as a shift.
4. The preconditioner (TPA) is not shift-invariant, and S(z) is complex-symmetric-not-Hermitian
   for η > 0, hence GMRES rather than CG.
5. The −ω branch uses z = −ω + iη, and the response combines conj(⟨·|dψ₋⟩); this is the
   downstream dielectric layer and is untouched.

Phase 1 implements only the dense/matrix-free identity with random Hermitian A, D.
Next step: block rational Krylov compression of (D − zI)⁻¹ on range(B†).
