
# GitHub Copilot プロンプト集 — Fe2MnAl + adaptive対応版

## Prompt 1: 物理層 + adaptive

You are extending gkemlin/response-calculations-metals to dynamic.
We have SchurModel(A,D,B) with S(z)=A-zI-B(D-zI)^{-1}B†, z=ω+iη.
Existing: src/operators/*, reduction/* from ShiftSchur_all_julia.txt

Create src/response/:
- occupation.jl: Fermi-Dirac f, df/dε (Cances Eq 2.5)
- dynamic_schur.jl: apply_schur_dynamic, solve_schur_dynamic, grid solver
- chi0.jl: chi0_sos (Adler-Wiser like QE epsilon.x) and chi0_schur (via solve_schur + δf term Eq 3.12)
- epsilon.jl: v_G=4π/|q+G|^2, epsilon_sos, epsilon_reduced
- adaptive_choice_extra_bands_dynamic.jl: dynamic version of run_adaptive_algorithm.jl Section 6 Appendix B.
  Static: extra bands chosen by gap/σ. Dynamic: need ω-dependent extra bands.
  Implement adaptive_extra_bands_dynamic(M,ωgrid,η,b; tol,s_max) using greedy_rational_shifts,
  and adaptive_choice_vs_default_comparison (default 4 shifts vs adaptive).
  Return AdaptiveExtraBandsResult(n_extra,k_dim,shifts,max_err,history).
Keep 1:1 notation with Cances paper. Only LinearAlgebra.

## Prompt 2: metal/Al40 + Fe2MnAl

Create metal/common/kubo.jl, metal/Al40/run_Al40.jl, metal/Fe2MnAl/run_Fe2MnAl.jl
kubo.jl: synthetic_metal_model(m,n,metal) where metal=:Al has 4 resonances -0.6,-0.1,0.35,0.8,
:Fe has -0.8,-0.2,0.4,0.9, :Fe2MnAl has 6 resonances -1.2,-0.6,-0.15,0.15,0.6,1.1 (Heusler, hard).
Use U=qr(randn).Q unitary, D=U Diag(λ) U', B=randn*Diag(w)*U' with w=0.6 inside [-1.2,1.2].
run_Al40.jl: ωgrid 0-10 eV 101 pts, ηs 0.2,0.1,0.05,0.02, call adaptive_extra_bands_dynamic, rational_compression_diagnostics, epsilon_sos.
run_Fe2MnAl.jl: ωgrid -3-3 eV 121 pts (Fermi focus), η same, s_max=16, show adaptive vs default comparison. Both runnable via julia --project=. metal/.../run_*.jl. Use MersenneTwister, check Q'Q-I.

## Prompt 3: ベンチマーク vs epsilon.x

Create src/benchmarks/vs_epsilon_x.jl and vs_sternheimer.jl
vs_epsilon_x: harness mimicking QE epsilon.x: benchmark_k_convergence, benchmark_eta_dependence. BenchRow(method,k_dim,eta,nshifts,Sigma_err,x_err). Use existing diagnostics.
vs_sternheimer: condition_number_estimate=(bandwidth+ω)/η, iteration_count ~ sqrt(κ)*log(1/tol). Show why shifted Sternheimer was strong in static (Cances) but weak in dynamic.

Refer to files in repo_dynamic_extension.zip, keep signatures from ShiftSchur_all_julia.txt.
