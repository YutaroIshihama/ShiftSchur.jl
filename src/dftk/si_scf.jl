module ShiftSchurDFTKSiSCF
using LinearAlgebra
import DFTK
import PseudoPotentialData
export SiSCFConfig, run_si_scf, scf_payload
Base.@kwdef struct SiSCFConfig
    lattice_constant_bohr::Float64 = 10.26
    Ecut::Float64 = 10.0
    kgrid::NTuple{3,Int} = (2,2,2)
    n_bands::Int = 8
    tol::Float64 = 1e-8
end
function run_si_scf(cfg::SiSCFConfig=SiSCFConfig())
    a=cfg.lattice_constant_bohr
    lattice=a/2 .* [0 1 1; 1 0 1; 1 1 0]
    psp=DFTK.PseudoFamily("dojo.nc.sr.pbe.v0_4_1.standard.upf")
    Si=DFTK.ElementPsp(:Si,psp)
    atoms=[Si,Si]; positions=[[0.0,0.0,0.0],[0.25,0.25,0.25]]
    model=DFTK.model_DFT(lattice,atoms,positions;functionals=[:gga_x_pbe,:gga_c_pbe])
    basis=DFTK.PlaneWaveBasis(model;Ecut=cfg.Ecut,kgrid=cfg.kgrid)
    # Keep the call simple and version-friendly. If the installed DFTK release
    # rejects n_bands/tol, adapt only this call and document the version.
    scfres=DFTK.self_consistent_field(basis;tol=cfg.tol,n_bands=cfg.n_bands)
    return scfres
end
function scf_payload(scfres)
    names=propertynames(scfres)
    req=(:basis,:ψ,:eigenvalues)
    missing=filter(x->!(x in names),req)
    isempty(missing) || error("DFTK SCF result is missing fields $(missing); installed API needs an adapter update")
    return (basis=getproperty(scfres,:basis),psi=getproperty(scfres,:ψ),eigenvalues=getproperty(scfres,:eigenvalues),occupation=(:occupation in names ? getproperty(scfres,:occupation) : nothing),rho=(:ρ in names ? getproperty(scfres,:ρ) : nothing),epsilon_F=(:εF in names ? getproperty(scfres,:εF) : nothing),raw=scfres)
end
end
