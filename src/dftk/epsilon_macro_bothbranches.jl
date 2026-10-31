"""Head-only macroscopic dielectric response with both frequency branches.

The Sternheimer and SoS paths use the same FFT normalization, perturbation,
occupations, k-point weights, q, Coulomb kernel, and branch convention.
The present implementation remains a same-k finite-q diagnostic; k->k+q is
not yet implemented.
"""
module ShiftSchurDFTKDielectricMacroBothBranches
using LinearAlgebra
using LinearMaps
using IterativeSolvers
import DFTK
import FFTW

export epsilon_macro_bothbranches, compare_sternheimer_sos, sos_band_convergence
export coefficients_to_real, real_to_coefficients

function _pw_to_real_scale(basis)
    N = prod(basis.fft_size)
    Omega = abs(det(basis.model.lattice))
    dV = Omega / N
    sqrt(N / dV)
end

function coefficients_to_real(basis, kpt, c::AbstractVector)
    length(c) == length(kpt.G_vectors) || throw(DimensionMismatch())
    grid = zeros(ComplexF64, basis.fft_size...)
    for (i, G) in enumerate(kpt.G_vectors)
        idx = ntuple(a -> mod(Int(G[a]), basis.fft_size[a]) + 1, 3)
        grid[idx...] = c[i]
    end
    FFTW.ifft(grid) * _pw_to_real_scale(basis)
end

function real_to_coefficients(basis, kpt, field::AbstractArray)
    size(field) == Tuple(basis.fft_size) || throw(DimensionMismatch())
    grid = FFTW.fft(field) / _pw_to_real_scale(basis)
    c = zeros(ComplexF64, length(kpt.G_vectors))
    for (i, G) in enumerate(kpt.G_vectors)
        idx = ntuple(a -> mod(Int(G[a]), basis.fft_size[a]) + 1, 3)
        c[i] = grid[idx...]
    end
    c
end

function build_phase_q(basis, q_red)
    nx, ny, nz = basis.fft_size
    phase = zeros(ComplexF64, nx, ny, nz)
    for ix in 0:nx-1, iy in 0:ny-1, iz in 0:nz-1
        phase[ix+1, iy+1, iz+1] = exp(2pi*im*(q_red[1]*ix/nx + q_red[2]*iy/ny + q_red[3]*iz/nz))
    end
    phase
end

function solve_dynamic_gmres(Hk, psi_occ, eps_n, rhs_n, z::Number;
                             reltol=1e-10, abstol=1e-12, restart=80, maxiter=400)
    n_pw, n_occ = size(psi_occ)
    px=zeros(ComplexF64,n_pw); qx=similar(px); hqx=similar(px)
    tmp=similar(px); ptmp=similar(px); coeff=zeros(ComplexF64,n_occ)
    function project_P!(y,x)
        mul!(coeff,adjoint(psi_occ),x); mul!(y,psi_occ,coeff); y
    end
    function apply_A!(y,x)
        project_P!(px,x); @. qx=x-px; mul!(hqx,Hk,qx)
        @. tmp=hqx-(eps_n+z)*qx; project_P!(ptmp,tmp)
        @. y=tmp-ptmp+px; y
    end
    project_P!(px,rhs_n); b=ComplexF64.(-rhs_n .+ px); bnorm=norm(b)
    bnorm < 1e-14 && return zeros(ComplexF64,n_pw)
    A=LinearMap{ComplexF64}(apply_A!,n_pw,n_pw;ismutating=true)
    x=zeros(ComplexF64,n_pw)
    x,hist=gmres!(x,A,b;restart=min(restart,n_pw),maxiter=maxiter,reltol=reltol,abstol=abstol,log=true)
    Ax=similar(x); apply_A!(Ax,x); relres=norm(Ax-b)/bnorm
    project_P!(px,x); ortherr=norm(px)/max(norm(x),eps(Float64)); target=max(reltol,abstol/bnorm)
    relres <= 10target || error("GMRES failed: relres=$relres target=$target z=$z")
    ortherr <= 1e-8 || error("GMRES left Q space: ortherr=$ortherr z=$z")
    @debug "DYNAMIC_GMRES" z relres ortherr hist
    x .- px
end

function _setup(scfres,q_red)
    basis=scfres.basis; Omega=abs(det(basis.model.lattice)); dV=Omega/prod(basis.fft_size)
    recip=2pi*inv(basis.model.lattice)'; q_cart=recip*q_red; q_norm=norm(q_cart)
    vc=4pi/(q_norm^2+1e-12); phase=build_phase_q(basis,q_red)
    (;basis,Omega,dV,vc,phase,phase_minus=conj.(phase),q_norm)
end

"""One resonant Sternheimer branch χ+(ω+iη)."""
function _chi_plus_sternheimer(scfres,omega,eta,q_red,setup)
    (;basis,Omega,dV,phase,phase_minus)=setup
    psi=scfres.ψ; eig=scfres.eigenvalues; occ=[DFTK.to_cpu(o) for o in scfres.occupation]
    rho=zeros(ComplexF64,basis.fft_size...); z=omega+im*eta
    for ik in eachindex(basis.kpoints)
        kpt=basis.kpoints[ik]; occk=occ[ik]; occ_idx=findall(>(0.1),occk)
        isempty(occ_idx) && continue
        psi_occ=psi[ik][:,occ_idx]; Hk=scfres.ham.blocks[ik]
        wk=basis.kweights[ik]
        for n in occ_idx
            psi_n_r=coefficients_to_real(basis,kpt,psi[ik][:,n])
            rhs_G=real_to_coefficients(basis,kpt,phase.*psi_n_r)
            x=solve_dynamic_gmres(Hk,psi_occ,eig[ik][n],rhs_G,z)
            rho .+= wk*occk[n].*conj.(psi_n_r).*coefficients_to_real(basis,kpt,x)
        end
    end
    sum(phase_minus.*rho)*dV/Omega
end

"""Truncated-band SoS for the same single branch and same-k perturbation."""
function _chi_plus_sos(scfres,omega,eta,q_red,setup; nempty::Union{Nothing,Int}=nothing)
    (;basis,Omega,phase)=setup
    psi=scfres.ψ; eig=scfres.eigenvalues; occ=[DFTK.to_cpu(o) for o in scfres.occupation]
    z=omega+im*eta; chi=0.0im
    for ik in eachindex(basis.kpoints)
        kpt=basis.kpoints[ik]; occk=occ[ik]; occ_idx=findall(>(0.1),occk); emp_idx=findall(<=(0.1),occk)
        nempty === nothing || (emp_idx = emp_idx[1:min(nempty,length(emp_idx))])
        wk=basis.kweights[ik]
        for n in occ_idx
            psi_n_r=coefficients_to_real(basis,kpt,psi[ik][:,n])
            rhs_G=real_to_coefficients(basis,kpt,phase.*psi_n_r)
            for m in emp_idx
                M=dot(psi[ik][:,m],rhs_G)
                chi += wk*occk[n]*abs2(M)/(eig[ik][n]-eig[ik][m]+z)
            end
        end
    end
    chi/Omega
end

"""Return both-branch Sternheimer and matched truncated-band SoS spectra.

χfull(ω)=χ+(ω)+conj(χ+(-ω)); this enforces χfull(-ω)=conj(χfull(ω))
and Im χfull(0)=0 for the present same-k diagnostic.
"""
function compare_sternheimer_sos(scfres,omegas::AbstractVector;eta=0.05,q_red=[0.01,0,0])
    setup=_setup(scfres,q_red)
    @info "Both-branch comparison" q_red setup.q_norm setup.vc setup.Omega
    chist=ComplexF64[]; chis=ComplexF64[]
    for omega in omegas
        sp=_chi_plus_sternheimer(scfres,omega,eta,q_red,setup)
        sm=_chi_plus_sternheimer(scfres,-omega,eta,q_red,setup)
        qp=_chi_plus_sos(scfres,omega,eta,q_red,setup)
        qm=_chi_plus_sos(scfres,-omega,eta,q_red,setup)
        sf=sp+conj(sm); qf=qp+conj(qm)
        push!(chist,sf); push!(chis,qf)
        @info "omega=$omega" chi_stern=sf chi_sos=qf eps_stern=1-setup.vc*sf eps_sos=1-setup.vc*qf
    end
    epsst=1 .- setup.vc.*chist; epss=1 .- setup.vc.*chis
    relchi=[abs(a-b)/max(abs(b),eps(Float64)) for (a,b) in zip(chist,chis)]
    releps=[abs(a-b)/max(abs(b),eps(Float64)) for (a,b) in zip(epsst,epss)]
    (;omegas=collect(omegas),eta,q_red,chi_sternheimer=chist,chi_sos=chis,
      epsilon_sternheimer=epsst,epsilon_sos=epss,relative_chi_error=relchi,
      relative_epsilon_error=releps,vc=setup.vc)
end

"""SoS convergence versus the number of empty bands, with one fixed SCF result.

The Sternheimer reference is evaluated once.  SoS is then truncated to the
lowest `nempty` empty states at each k point, avoiding repeated SCF runs.
"""
function sos_band_convergence(scfres, omegas::AbstractVector;
                              eta=0.05, q_red=[0.01,0,0],
                              nempty_list=[2,4,8,12,16,24])
    setup=_setup(scfres,q_red)
    occ=[DFTK.to_cpu(o) for o in scfres.occupation]
    available=minimum(count(<=(0.1),o) for o in occ)
    caps=sort(unique(min.(Int.(collect(nempty_list)),available)))
    filter!(>(0),caps)
    isempty(caps) && error("No empty bands are available in the SCF result")
    @info "SoS empty-band convergence" available caps q_red eta

    # Expensive matrix-free reference only once per +/- frequency.
    chist=ComplexF64[]
    for omega in omegas
        sp=_chi_plus_sternheimer(scfres,omega,eta,q_red,setup)
        sm=_chi_plus_sternheimer(scfres,-omega,eta,q_red,setup)
        push!(chist,sp+conj(sm))
    end
    epsst=1 .- setup.vc.*chist

    rows=NamedTuple[]
    for ne in caps
        chis=ComplexF64[]
        for omega in omegas
            qp=_chi_plus_sos(scfres,omega,eta,q_red,setup;nempty=ne)
            qm=_chi_plus_sos(scfres,-omega,eta,q_red,setup;nempty=ne)
            push!(chis,qp+conj(qm))
        end
        epss=1 .- setup.vc.*chis
        chierr=[abs(a-b)/max(abs(b),eps(Float64)) for (a,b) in zip(chis,chist)]
        epserr=[abs(a-b)/max(abs(b),eps(Float64)) for (a,b) in zip(epss,epsst)]
        push!(rows,(nempty=ne,chi_sos=chis,epsilon_sos=epss,
                    max_relative_chi_error=maximum(chierr),
                    max_relative_epsilon_error=maximum(epserr),
                    static_epsilon=epss[findfirst(iszero,omegas) === nothing ? 1 : findfirst(iszero,omegas)]))
    end
    (;omegas=collect(omegas),eta,q_red,available_empty_bands=available,
      chi_sternheimer=chist,epsilon_sternheimer=epsst,rows,vc=setup.vc)
end

epsilon_macro_bothbranches(args...;kwargs...)=compare_sternheimer_sos(args...;kwargs...)
end
