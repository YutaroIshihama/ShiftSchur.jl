"""Mesh-compatible k -> k+q both-branch Sternheimer/SoS diagnostic.

q must map every stored k point to another stored k point of the same spin,
modulo a reciprocal lattice vector. This first implementation deliberately
rejects arbitrary q and symmetry-reduced meshes instead of silently falling
back to the incorrect same-k approximation.
"""
module ShiftSchurDFTKDielectricKplusQ
using LinearAlgebra
using LinearMaps
using IterativeSolvers
import DFTK
import FFTW
export compare_kplusq_sternheimer_sos, build_kplusq_map

function _scale(basis)
    N=prod(basis.fft_size); dV=abs(det(basis.model.lattice))/N; sqrt(N/dV)
end
function coeffs_to_real(basis,kpt,c)
    g=zeros(ComplexF64,basis.fft_size...)
    for (i,G) in enumerate(kpt.G_vectors)
        idx=ntuple(a->mod(Int(G[a]),basis.fft_size[a])+1,3); g[idx...]=c[i]
    end
    FFTW.ifft(g)*_scale(basis)
end
function real_to_coeffs(basis,kpt,f)
    g=FFTW.fft(f)/_scale(basis); c=zeros(ComplexF64,length(kpt.G_vectors))
    for (i,G) in enumerate(kpt.G_vectors)
        idx=ntuple(a->mod(Int(G[a]),basis.fft_size[a])+1,3); c[i]=g[idx...]
    end
    c
end
function reciprocal_phase(basis,Gwrap)
    nx,ny,nz=basis.fft_size; p=zeros(ComplexF64,nx,ny,nz)
    for ix in 0:nx-1,iy in 0:ny-1,iz in 0:nz-1
        p[ix+1,iy+1,iz+1]=exp(2pi*im*(Gwrap[1]*ix/nx+Gwrap[2]*iy/ny+Gwrap[3]*iz/nz))
    end
    p
end
function _mod1center(x)
    x-round(x)
end
"""Return `(target, Gwrap)` for each source k, where k+q=k_target+Gwrap."""
function build_kplusq_map(basis,q_red;tol=1e-8)
    kpts=basis.kpoints; result=NamedTuple[]
    for (ik,k) in enumerate(kpts)
        wanted=collect(k.coordinate).+q_red; found=nothing
        for (jk,kq) in enumerate(kpts)
            k.spin==kq.spin || continue
            delta=wanted.-collect(kq.coordinate); Gwrap=round.(Int,delta)
            if maximum(abs.(delta.-Gwrap))<tol
                found=(source=ik,target=jk,Gwrap=Gwrap); break
            end
        end
        found===nothing && error("q=$q_red does not map stored k[$ik]=$(collect(k.coordinate)) to a stored k point. Use a full, unreduced mesh and mesh-compatible q, e.g. q=(1/Nx,0,0).")
        push!(result,found)
    end
    result
end
function solve_gmres(H,psi_occ,eps_n,rhs,z;reltol=1e-10,abstol=1e-12,restart=80,maxiter=400)
    n,nocc=size(psi_occ); px=zeros(ComplexF64,n); qx=similar(px); hqx=similar(px); tmp=similar(px); ptmp=similar(px); coeff=zeros(ComplexF64,nocc)
    function P!(y,x); mul!(coeff,adjoint(psi_occ),x); mul!(y,psi_occ,coeff); y; end
    function A!(y,x); P!(px,x); @. qx=x-px; mul!(hqx,H,qx); @. tmp=hqx-(eps_n+z)*qx; P!(ptmp,tmp); @. y=tmp-ptmp+px; y; end
    P!(px,rhs); b=ComplexF64.(-rhs.+px); nb=norm(b); nb<1e-14 && return zeros(ComplexF64,n)
    A=LinearMap{ComplexF64}(A!,n,n;ismutating=true); x=zeros(ComplexF64,n)
    x,h=gmres!(x,A,b;restart=min(restart,n),maxiter,reltol,abstol,log=true)
    Ax=similar(x); A!(Ax,x); rr=norm(Ax-b)/nb; P!(px,x); oe=norm(px)/max(norm(x),eps(Float64)); target=max(reltol,abstol/nb)
    rr<=10target || error("k+q GMRES failed: relres=$rr target=$target z=$z")
    oe<=1e-8 || error("k+q GMRES left Q space: ortherr=$oe z=$z")
    x.-px
end
function _setup(scfres,q)
    basis=scfres.basis; Omega=abs(det(basis.model.lattice)); recip=2pi*inv(basis.model.lattice)'; qnorm=norm(recip*q); vc=4pi/(qnorm^2+1e-12)
    (;basis,Omega,vc,qnorm,kmap=build_kplusq_map(basis,q))
end
function _rhs_at_target(setup,psi,ik,n)
    basis=setup.basis; entry=setup.kmap[ik]; ks=basis.kpoints[ik]; kt=basis.kpoints[entry.target]
    us=coeffs_to_real(basis,ks,psi[ik][:,n]); phase=reciprocal_phase(basis,entry.Gwrap)
    real_to_coeffs(basis,kt,phase.*us)
end
function _chi_plus_stern(scfres,omega,eta,setup)
    basis=setup.basis; psi=scfres.ψ; eig=scfres.eigenvalues; occ=[DFTK.to_cpu(o) for o in scfres.occupation]; chi=0.0im; z=omega+im*eta
    for entry in setup.kmap
        ik=entry.source; jk=entry.target; occn=findall(>(0.1),occ[ik]); occq=findall(>(0.1),occ[jk]); psioccq=psi[jk][:,occq]
        for n in occn
            rhs=_rhs_at_target(setup,psi,ik,n); x=solve_gmres(scfres.ham.blocks[jk],psioccq,eig[ik][n],rhs,z)
            chi+=basis.kweights[ik]*occ[ik][n]*dot(rhs,x)/setup.Omega
        end
    end
    chi
end
function _chi_plus_sos(scfres,omega,eta,setup;nempty=nothing)
    basis=setup.basis; psi=scfres.ψ; eig=scfres.eigenvalues; occ=[DFTK.to_cpu(o) for o in scfres.occupation]; chi=0.0im; z=omega+im*eta
    for entry in setup.kmap
        ik=entry.source; jk=entry.target; occn=findall(>(0.1),occ[ik]); empty=findall(<=(0.1),occ[jk]); nempty===nothing || (empty=empty[1:min(nempty,length(empty))])
        for n in occn
            rhs=_rhs_at_target(setup,psi,ik,n)
            for m in empty
                M=dot(psi[jk][:,m],rhs); chi+=basis.kweights[ik]*occ[ik][n]*abs2(M)/(eig[ik][n]-eig[jk][m]+z)/setup.Omega
            end
        end
    end
    chi
end
function compare_kplusq_sternheimer_sos(scfres,omegas;eta=0.05,q_red=[0.5,0,0],nempty=nothing)
    setup=_setup(scfres,q_red); @info "k->k+q comparison" q_red setup.qnorm setup.vc
    cs=ComplexF64[]; cq=ComplexF64[]
    for w in omegas
        sp=_chi_plus_stern(scfres,w,eta,setup); sm=_chi_plus_stern(scfres,-w,eta,setup)
        qp=_chi_plus_sos(scfres,w,eta,setup;nempty); qm=_chi_plus_sos(scfres,-w,eta,setup;nempty)
        push!(cs,sp+conj(sm)); push!(cq,qp+conj(qm))
    end
    es=1 .-setup.vc.*cs; eq=1 .-setup.vc.*cq
    (;omegas=collect(omegas),q_red,eta,chi_sternheimer=cs,chi_sos=cq,epsilon_sternheimer=es,epsilon_sos=eq,
      relative_chi_error=[abs(a-b)/max(abs(b),eps()) for (a,b) in zip(cs,cq)],kmap=setup.kmap,vc=setup.vc)
end
end
