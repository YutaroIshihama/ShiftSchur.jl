using Test
using Printf
using ShiftSchur
include(joinpath(@__DIR__,"..","src","dftk","si_scf.jl"))
include(joinpath(@__DIR__,"..","src","dftk","epsilon_macro_bothbranches.jl"))
using .ShiftSchurDFTKSiSCF
using .ShiftSchurDFTKDielectricMacroBothBranches

@testset "Si SoS empty-band convergence against both-branch Sternheimer" begin
    cfg=SiSCFConfig(n_bands=30)
    scfres=run_si_scf(cfg)
    omegas=[0.0,0.1,0.2,0.3,0.4]
    result=sos_band_convergence(scfres,omegas;eta=0.05,q_red=[0.01,0.0,0.0],nempty_list=[2,4,8,12,16,22,26])

    println("\nnempty  Re eps(0)  max rel chi  max rel eps")
    for row in result.rows
        @printf("%6d  %9.6f  %11.3e  %11.3e\n",row.nempty,real(row.static_epsilon),row.max_relative_chi_error,row.max_relative_epsilon_error)
    end

    output=joinpath(@__DIR__,"..","docs","sos_band_convergence.csv")
    open(output,"w") do io
        println(io,"nempty,omega,Re_chi_sos,Im_chi_sos,Re_eps_sos,Im_eps_sos,Re_chi_stern,Im_chi_stern,Re_eps_stern,Im_eps_stern,rel_chi_err,rel_eps_err")
        for row in result.rows, i in eachindex(result.omegas)
            cs=row.chi_sos[i]; es=row.epsilon_sos[i]
            ct=result.chi_sternheimer[i]; et=result.epsilon_sternheimer[i]
            erc=abs(cs-ct)/max(abs(ct),eps(Float64)); ere=abs(es-et)/max(abs(et),eps(Float64))
            println(io,"$(row.nempty),$(result.omegas[i]),$(real(cs)),$(imag(cs)),$(real(es)),$(imag(es)),$(real(ct)),$(imag(ct)),$(real(et)),$(imag(et)),$erc,$ere")
        end
    end

    @test length(result.rows)>=2
    @test all(isfinite(row.max_relative_chi_error) for row in result.rows)
    @test all(isfinite(row.max_relative_epsilon_error) for row in result.rows)
    # Robust convergence criterion: the largest band set must improve over the smallest.
    @test result.rows[end].max_relative_chi_error < result.rows[1].max_relative_chi_error
    @test result.rows[end].max_relative_epsilon_error < result.rows[1].max_relative_epsilon_error
    @test abs(imag(result.rows[end].static_epsilon)) < 1e-12
    println("Saved $output")
end
