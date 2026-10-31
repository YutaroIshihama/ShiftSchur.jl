using Test
using Printf
using ShiftSchur
include(joinpath(@__DIR__,"..","src","dftk","si_scf.jl"))
include(joinpath(@__DIR__,"..","src","dftk","epsilon_kplusq_bothbranches.jl"))
using .ShiftSchurDFTKSiSCF
using .ShiftSchurDFTKDielectricKplusQ

@testset "mesh-compatible k->k+q Sternheimer versus SoS" begin
    # q=(1/2,0,0) is compatible with a full 2x2x2 mesh. The SCF helper must
    # retain the full mesh (disable symmetry reduction if it currently does not).
    scfres = run_si_scf(
    SiSCFConfig(
        n_bands=30,
        use_symmetries=false,
    ),
)
@test length(scfres.basis.kpoints) == 8

println("full-mesh k points:")
for (ik, kpt) in enumerate(scfres.basis.kpoints)
    println(
        "  k[$ik] = ",
        collect(kpt.coordinate),
        " weight=",
        scfres.basis.kweights[ik],
    )
end    
    q=[0.5,0.0,0.0]; omegas=[0.0,0.1,0.2]
    mapping=build_kplusq_map(scfres.basis,q)
    @test length(mapping)==length(scfres.basis.kpoints)
    @test all(e->e.target in eachindex(scfres.basis.kpoints),mapping)
    result=compare_kplusq_sternheimer_sos(scfres,omegas;eta=0.05,q_red=q,nempty=26)
    println("omega Re eps(St) Im eps(St) Re eps(SoS) Im eps(SoS) rel chi")
    for i in eachindex(omegas)
        a=result.epsilon_sternheimer[i]; b=result.epsilon_sos[i]
        @printf("%.2f %.8f %.8f %.8f %.8f %.3e\n",omegas[i],real(a),imag(a),real(b),imag(b),result.relative_chi_error[i])
    end
    @test all(isfinite,real.(result.epsilon_sternheimer))
    @test all(isfinite,imag.(result.epsilon_sternheimer))
    @test abs(imag(result.epsilon_sternheimer[1]))<1e-12
    @test maximum(result.relative_chi_error)<0.25
end
