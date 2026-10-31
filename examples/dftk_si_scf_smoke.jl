include(joinpath(@__DIR__, "..", "src", "dftk", "si_scf.jl"))
using .ShiftSchurDFTKSiSCF
using DFTK

cfg = SiSCFConfig()
println("Running DFTK Si SCF: Ecut=$(cfg.Ecut) Ha, kgrid=$(cfg.kgrid)")
scfres = run_si_scf(cfg)
p = scf_payload(scfres)
println("SCF OK")
println("pkgversion(DFTK) = ", pkgversion(DFTK))
println("scfres.εF = ", scfres.εF)
println("length(scfres.basis.kpoints) = ", length(scfres.basis.kpoints))
println("number of wavefunction blocks = ", length(p.psi))
println("size(scfres.ψ[1]) = ", size(scfres.ψ[1]))
println("first eigenvalue block = ", p.eigenvalues[1])
