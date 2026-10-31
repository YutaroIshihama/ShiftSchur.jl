# DFTK-Si response adapter patch

This removes the intentional `DFTKResponseNotImplemented` stop and replaces it with a real DFTK static-response bridge.

The physical oracle is `DFTK.apply_χ0(scfres, δV)`, not a guessed occupied/virtual KS partition. A small periodic density-probe basis is sampled, producing a finite static χ0 matrix. A Hermitian pole representation is fitted so that `Σ(0) = -χ0` on this sampled subspace.

**Important:** this is strictly a static correctness bridge. `metadata.static_only == true`. It makes the Si end-to-end smoke test executable and provides the right seam for ShiftSchur, but it does not claim that the fitted pole dependence at nonzero frequency is the physical DFTK dynamic response.

Run:

    julia --project=. examples/dftk_si_scf_smoke.jl
    julia --project=. examples/dftk_si_shift_schur.jl
    julia --project=. -e 'using Pkg; Pkg.test()'

Next step: expose/implement DFTK's frequency-dependent Sternheimer apply and replace the static pole fit with that dynamic oracle before producing epsilon(omega).
