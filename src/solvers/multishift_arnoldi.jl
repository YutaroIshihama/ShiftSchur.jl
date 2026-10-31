using LinearAlgebra

"""One unrestarted Arnoldi basis for H; solve (H-z_j I)x_j=b for all z_j."""
function multishift_arnoldi(H,b,zs; krylovdim=min(length(b),80), tol=1e-10)
    N=length(b); m=min(krylovdim,N); beta=norm(b)
    beta==0 && return [zeros(promote_type(eltype(b),ComplexF64),N) for _ in zs], (;iterations=0,residuals=zeros(length(zs)))
    T=promote_type(eltype(H),eltype(b),eltype(zs),ComplexF64)
    V=zeros(T,N,m+1); K=zeros(T,m+1,m); V[:,1].=b/beta; w=zeros(T,N); kdone=m
    for j in 1:m
        mul!(w,H,@view(V[:,j]))
        for i in 1:j
            K[i,j]=dot(@view(V[:,i]),w); w .-= K[i,j].*@view(V[:,i])
        end
        K[j+1,j]=norm(w)
        if abs(K[j+1,j]) <= 10 * eps(Float64)
            kdone=j; break
        end
        V[:,j+1].=w/K[j+1,j]
    end
    Vk=@view V[:,1:kdone]; Kk=@view K[1:kdone+1,1:kdone]
    rhs=zeros(T,kdone+1); rhs[1]=beta
    xs=Vector{Vector{T}}(undef,length(zs)); residuals=zeros(Float64,length(zs))
    for (q,z) in pairs(zs)
        M=Matrix(Kk); M[1:kdone,1:kdone] .-= z*I(kdone)
        y=M\rhs; xs[q]=Vector(Vk*y); residuals[q]=norm(M*y-rhs)/beta
    end
    xs,(;iterations=kdone,residuals=residuals,converged=residuals .<= tol)
end
