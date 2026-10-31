# ShiftSchur.jl
Restoring Shift Invariance in Dynamical Schur-Complement Sternheimer Response

## screen shot ps1

`$out="ShiftSchur_all_julia.txt"; Remove-Item $out -ErrorAction SilentlyContinue; Get-ChildItem -Path .\src,.\test -Recurse -Filter *.jl | Sort-Object FullName | ForEach-Object { "===== FILE: $($_.FullName.Substring((Get-Location).Path.Length+1)) =====`r`n"; Get-Content $_.FullName -Raw; "`r`n`r`n" } | Set-Content $out -Encoding utf8`

## Logs on Oct. 28

これがそのままSISCの Figure 1,2,3 になる。

- convergence_eta.png : k が増えると指数的に誤差が落ちる、ηが小さいと難しい
- rational_vs_poly.png : 同じkで rational が 1e5 倍良い
- greedy.png : greedy が等間隔より速く落ちる

# ShiftSchur DFTK-Si end-to-end scaffold

This bundle is intended to be copied into the ShiftSchur repository. It deliberately separates a **DFTK SCF smoke test that should be runnable now** from the **true Sternheimer/response adapter**, whose implementation depends on the exact installed DFTK API and must not be replaced by a fabricated dense band-space model.

## Goal
1. Run a small silicon SCF calculation in DFTK.
2. Collect the converged basis, orbitals, eigenvalues, density and Fermi information in a version-tolerant container.
3. Hand these data to `build_si_response_problem`, the single integration seam to be completed against the installed DFTK version.
4. Construct ShiftSchur `SchurModel`, adaptive rational ROM and compare exact/reduced response on an omega grid.

## Quick start
From the ShiftSchur repository root:

    julia --project=. -e 'using Pkg; Pkg.add(["DFTK","PseudoPotentialData"]); Pkg.instantiate()'
    julia --project=. examples/dftk_si_scf_smoke.jl
    julia --project=. examples/dftk_si_shift_schur.jl

The second script is expected to stop with an explicit `DFTKResponseNotImplemented` error until `src/dftk/si_response_adapter.jl` is connected to the installed DFTK Sternheimer / response API. This is intentional: no fake A/B/D matrices are silently substituted for the physical response problem.

## Files
- `examples/dftk_si_scf_smoke.jl`: real silicon SCF smoke test.
- `src/dftk/si_scf.jl`: silicon model and SCF helper.
- `src/dftk/si_response_adapter.jl`: strict adapter boundary, with validation.
- `examples/dftk_si_shift_schur.jl`: end-to-end driver after adapter completion.
- `test/test_dftk_si_adapter.jl`: offline validation tests not requiring a DFTK download.
- `docs/GITHUB_COPILOT_DFTK_SI_PROMPT.md`: implementation prompt.

## Important physics constraint
Do not identify ordinary Kohn-Sham occupied/virtual eigenvalue blocks with the Cancès response Schur blocks merely because both admit a matrix partition. The physical `A`, `B`, `D` must come from the chosen linear-response/Sternheimer formulation, including projectors, occupations, perturbation convention, and the requested response observable.
