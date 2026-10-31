# GitHub Copilot prompt: complete the real DFTK-Si response adapter

We now want the first real-material end-to-end validation of ShiftSchur using silicon and the installed DFTK release.

## Non-negotiable constraints
- Preserve all currently passing ShiftSchur tests and sign convention `z = omega + i*eta`, eliminated resolvent `(D-zI)^(-1)`.
- Do not fabricate `A`, `B`, `D` from an arbitrary partition of a Kohn-Sham eigenvalue matrix. They must be the blocks of the actual DFTK linear-response/Sternheimer formulation being reduced.
- Do not silently fall back to a toy model if DFTK APIs are unavailable. Fail loudly with an actionable error.
- Keep the first run deliberately small: Si primitive cell, low Ecut and 2x2x2 k mesh are a smoke test, not a converged spectrum.
- Record exact DFTK and Julia versions and document every DFTK internal API touched.

## First inspect the installed source
Using the active Julia project, locate DFTK source methods relevant to:
1. `self_consistent_field`
2. Hamiltonian construction / `HamiltonianBlock`
3. occupied-state projectors and occupations
4. Sternheimer or `apply_χ0` / susceptibility / polarizability routines
5. density/potential response and q=0 handling

Do not guess function names. Use `methods`, `@which`, `pathof(DFTK)`, and source inspection. Prefer public APIs; if internal APIs are unavoidable isolate them in `src/dftk/si_response_adapter.jl`.

## Implement in this order
1. Make `examples/dftk_si_scf_smoke.jl` run on the installed DFTK version. If `PseudoFamily` belongs to DFTK versus PseudoPotentialData in this version, use the actually supported namespace.
2. Implement a version-tolerant `scf_payload` extracting basis, wavefunctions, eigenvalues, occupations/density/Fermi level when available.
3. Determine the precise response space for a q=0 density-head perturbation. State its dimensions and index convention in code comments.
4. Construct a matrix-free reference apply for the dynamic Sternheimer response at `z=omega+i*eta`.
5. For only the tiny Si smoke configuration, materialize the relevant response operator/blocks to obtain a dense reference `SchurModel(A,D,B)` for verification. This is for correctness validation only, not production scaling.
6. Add checks that `A` and `D` are Hermitian to numerical tolerance, `B` has compatible dimensions, and the full auxiliary block `[A B; B' D]` reproduces the direct reference response.
7. Connect the existing adaptive rational Krylov code without changing it.
8. Compare exact dense Schur/self-energy action and reduced action over an omega grid. Save CSV with omega, eta, exact/reduced response and relative error.
9. Add an end-to-end integration test gated by an environment variable such as `SHIFTSCHUR_RUN_DFTK_INTEGRATION=1`, so ordinary `Pkg.test()` remains fast/offline while CI or a workstation can opt into real DFTK-Si.
10. Only after correctness passes, replace dense materialization of D with a LinearMap/matrix-free shifted-solve adapter. Keep the dense route as a reference oracle.

## Acceptance criteria
- normal `Pkg.test()` stays green without network or pseudopotential download;
- opt-in DFTK integration SCF converges and reports its DFTK version;
- exact Schur/full-auxiliary identity is <=1e-9 on the materialized tiny response problem;
- reduced response error decreases as rational poles are added;
- adaptive stopping reaches its requested tolerance or explicitly reports the pole cap;
- no toy/synthetic matrix is used in the real Si integration path;
- output states that Ecut=10 Ha and 2x2x2 k grid are smoke-test parameters, not physically converged values.

## Deliverable
Commit the DFTK adapter, the opt-in integration test, and a short `docs/dftk_si.md` documenting equations, block definitions, DFTK source locations, index conventions, units, q convention, and reproduction command. Do not modify the core ShiftSchur algorithms unless a failing mathematical identity demonstrates a core bug.
