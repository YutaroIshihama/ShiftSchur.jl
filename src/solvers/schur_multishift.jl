using LinearAlgebra

"""Solve [A-zI-B(D-zI)^-1B']x=b at all z using one auxiliary Krylov basis."""
function schur_multishift(A,B,D,b,zs; krylovdim=min(size(A,1)+size(D,1),80),tol=1e-10)
    H=AuxiliaryLinearization(A,B,D)
    Xaux,info=multishift_arnoldi(H,lift_rhs(H,b),collect(zs);krylovdim=krylovdim,tol=tol)
    X=[Vector(physical_part(H,x)) for x in Xaux]
    X,info
end
