module ShiftSchurDFTKSiSCF

using LinearAlgebra
using DFTK
using PseudoPotentialData

export SiSCFConfig, run_si_scf, scf_payload

Base.@kwdef struct SiSCFConfig
    lattice_constant_bohr::Float64 = 10.26
    Ecut::Float64 = 30.0
    kgrid::NTuple{3,Int} = (6, 6, 6)
    n_bands::Int = 60
    tol::Float64 = 1e-6
end

function run_si_scf(cfg::SiSCFConfig=SiSCFConfig())
    a = cfg.lattice_constant_bohr
    lattice = a / 2 .* [0 1 1; 1 0 1; 1 1 0]

    pd_family = PseudoFamily("dojo.nc.sr.lda.v0_4_1.standard.upf")
    Si = ElementPsp(:Si, pd_family)

    atoms = [Si, Si]
    positions = [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]

    model = model_DFT(
        lattice,
        atoms,
        positions;
        functionals=LDA(),
        temperature=0.01,
        smearing=Smearing.Gaussian(),
    )

    basis = PlaneWaveBasis(
        model;
        Ecut=cfg.Ecut,
        kgrid=collect(cfg.kgrid),
    )

    # Use FixedBands or AdaptiveBands to force n_bands
    # In DFTK 0.8, use nbandsalg keyword
    scfres = try
        # Try FixedBands
        nbandsalg = DFTK.FixedBands(n_bands=cfg.n_bands)
        self_consistent_field(basis; tol=cfg.tol, nbandsalg=nbandsalg)
    catch e1
        try
            # Try AdaptiveBands with n_bands_converge
            nbandsalg = DFTK.AdaptiveBands(model; n_bands_converge=cfg.n_bands)
            self_consistent_field(basis; tol=cfg.tol, nbandsalg=nbandsalg)
        catch e2
            @warn "Failed to set n_bands via nbandsalg: $e1 / $e2, using default"
            self_consistent_field(basis; tol=cfg.tol)
        end
    end

    return scfres
end

function scf_payload(scfres)
    return (
        basis=scfres.basis,
        psi=scfres.ψ,
        eigenvalues=scfres.eigenvalues,
        occupation=hasproperty(scfres, :occupation) ? scfres.occupation : nothing,
        rho=hasproperty(scfres, :ρ) ? scfres.ρ : nothing,
        epsilon_F=hasproperty(scfres, :εF) ? scfres.εF : nothing,
        raw=scfres,
    )
end

end
