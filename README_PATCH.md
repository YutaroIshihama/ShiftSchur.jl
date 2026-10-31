# ShiftSchur: auxiliary linearization + multi-shift restoration

This patch implements the exact linearization

S(z) = A - zI - B (D-zI)^(-1) B'  -->  Haux = [A B; B' D]

so that `(Haux-zI)[x;y]=[b;0]` has the same physical component `x` as `S(z)x=b`.
The key point is that all frequency dependence is now the scalar shift `-zI`, so one Krylov basis can be reused for a family of frequencies.

Files:
- `src/operators/auxiliary_linearization.jl`: matrix-free block operator and exact RHS/lift/project helpers.
- `src/solvers/multishift_arnoldi.jl`: one Arnoldi basis, small shifted least-squares solve for every complex shift.
- `src/solvers/schur_multishift.jl`: public driver returning physical Sternheimer solutions.
- `test/test_auxiliary_linearization.jl`: exact Schur-vs-auxiliary identity test.
- `test/test_multishift_schur.jl`: many-shift-vs-direct solve test.
- `test/test_silicon_second.jl`: adapter-style second Si regression. Hook it to the existing Si fixture.

Integration: include the three source files from `src/ShiftSchur.jl`; include the three tests from `test/runtests.jl`.
No assumption that D is diagonal is needed. For large production systems provide `mul!`-compatible A/B/D operators; the compact implementation below accepts ordinary matrices and adjoint(B).

Numerical note: the supplied Arnoldi solver is deliberately transparent and is a correctness/reference implementation. It uses a single unrestarted basis. Production code can replace only the Arnoldi backend with a restarted/deflated implementation without changing the auxiliary formulation.
