using Test
using Printf
using Dates
using ShiftSchur

include(joinpath(@__DIR__, "..", "src", "dftk", "si_scf.jl"))
include(joinpath(@__DIR__, "..", "src", "dftk", "epsilon_kplusq_bothbranches.jl"))
using .ShiftSchurDFTKSiSCF
using .ShiftSchurDFTKDielectricKplusQ

"""
Run the static, mesh-compatible q sequence q=(1/Nx,0,0).

The default sequence is intentionally small enough for a desktop CPU:
    (2,2,2) -> q=1/2
    (4,2,2) -> q=1/4

Set ENV["SHIFTSCHUR_QCONV_HEAVY"]="true" to append:
    (4,4,4) -> q=1/4
    (6,4,4) -> q=1/6

This is a finite-q convergence diagnostic, not yet a production q->0
extrapolation. Every calculation uses the full unreduced k mesh.
"""
function q_convergence_cases()
    cases = [
        (kgrid=(2, 2, 2), q_red=[0.5, 0.0, 0.0]),
        (kgrid=(4, 2, 2), q_red=[0.25, 0.0, 0.0]),
    ]
    if lowercase(get(ENV, "SHIFTSCHUR_QCONV_HEAVY", "false")) in ("1", "true", "yes")
        append!(cases, [
            (kgrid=(4, 4, 4), q_red=[0.25, 0.0, 0.0]),
            (kgrid=(6, 4, 4), q_red=[1 / 6, 0.0, 0.0]),
        ])
    end
    cases
end

@testset "mesh-compatible finite-q convergence" begin
    eta = 0.05
    omega = 0.0
    n_bands = 30
    nempty = 26
    rows = NamedTuple[]

    for case in q_convergence_cases()
        kgrid = case.kgrid
        q_red = case.q_red
        println("\n=== q-convergence case: kgrid=$kgrid q_red=$q_red ===")

        t0 = time_ns()
        scfres = run_si_scf(SiSCFConfig(
            kgrid=kgrid,
            n_bands=n_bands,
            use_symmetries=false,
        ))
        scf_seconds = (time_ns() - t0) / 1e9

        nk_expected = prod(kgrid)
        @test length(scfres.basis.kpoints) == nk_expected
        @test all(isapprox(w, 1 / nk_expected; atol=1e-14, rtol=1e-12)
                  for w in scfres.basis.kweights)

        mapping = build_kplusq_map(scfres.basis, q_red)
        @test length(mapping) == nk_expected
        @test all(entry -> entry.target in eachindex(scfres.basis.kpoints), mapping)
        @test length(unique(entry.target for entry in mapping)) == nk_expected

        t1 = time_ns()
        result = compare_kplusq_sternheimer_sos(
            scfres,
            [omega];
            eta=eta,
            q_red=q_red,
            nempty=nempty,
        )
        response_seconds = (time_ns() - t1) / 1e9

        eps_stern = only(result.epsilon_sternheimer)
        eps_sos = only(result.epsilon_sos)
        chi_stern = only(result.chi_sternheimer)
        chi_sos = only(result.chi_sos)
        rel_chi = only(result.relative_chi_error)

        @test isfinite(real(eps_stern)) && isfinite(imag(eps_stern))
        @test isfinite(real(eps_sos)) && isfinite(imag(eps_sos))
        @test abs(imag(eps_stern)) < 1e-12
        @test abs(imag(eps_sos)) < 1e-12
        @test rel_chi < 0.25

        push!(rows, (
            kgrid=kgrid,
            nk=nk_expected,
            qx=q_red[1],
            qnorm=sqrt(4pi / result.vc),
            epsilon_sternheimer=eps_stern,
            epsilon_sos=eps_sos,
            chi_sternheimer=chi_stern,
            chi_sos=chi_sos,
            relative_chi_error=rel_chi,
            scf_seconds=scf_seconds,
            response_seconds=response_seconds,
        ))
    end

    @test all(rows[i + 1].qx <= rows[i].qx for i in 1:length(rows)-1)

    println("\nkgrid       nk      qx       |q|       Re eps(St)  Re eps(SoS) rel chi   SCF[s]  response[s]")
    for row in rows
        @printf("(%d,%d,%d) %5d  %7.4f  %8.5f  %11.7f  %11.7f  %.3e  %7.2f  %10.2f\n",
                row.kgrid..., row.nk, row.qx, row.qnorm,
                real(row.epsilon_sternheimer), real(row.epsilon_sos),
                row.relative_chi_error, row.scf_seconds, row.response_seconds)
    end

    output = joinpath(@__DIR__, "..", "docs", "kplusq_q_convergence.csv")
    mkpath(dirname(output))
    open(output, "w") do io
        println(io, "timestamp,kx,ky,kz,nk,qx,qnorm,Re_chi_stern,Im_chi_stern,Re_chi_sos,Im_chi_sos,Re_eps_stern,Im_eps_stern,Re_eps_sos,Im_eps_sos,rel_chi_err,scf_seconds,response_seconds")
        stamp = Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS")
        for row in rows
            println(io, join((
                stamp, row.kgrid[1], row.kgrid[2], row.kgrid[3], row.nk,
                row.qx, row.qnorm,
                real(row.chi_sternheimer), imag(row.chi_sternheimer),
                real(row.chi_sos), imag(row.chi_sos),
                real(row.epsilon_sternheimer), imag(row.epsilon_sternheimer),
                real(row.epsilon_sos), imag(row.epsilon_sos),
                row.relative_chi_error, row.scf_seconds, row.response_seconds,
            ), ','))
        end
    end
    println("Saved $output")

    # Do not assert monotonic epsilon(q): anisotropic k meshes and finite-size
    # quadrature can make the first few points non-monotonic. The CSV is the
    # convergence diagnostic; strict extrapolation belongs to a denser sequence.
end
