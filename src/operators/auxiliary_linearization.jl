using LinearAlgebra

struct AuxiliaryLinearization{TA,TB,TD,T}
    A::TA
    B::TB
    D::TD
    nphys::Int
    naux::Int
    work_phys::Vector{T}
    work_aux::Vector{T}
end

function AuxiliaryLinearization(A, B, D; T=promote_type(eltype(A),eltype(B),eltype(D),ComplexF64))
    n,m=size(B)
    size(A)==(n,n) || throw(DimensionMismatch("A must be n x n"))
    size(D)==(m,m) || throw(DimensionMismatch("D must be m x m"))
    AuxiliaryLinearization{typeof(A),typeof(B),typeof(D),T}(A,B,D,n,m,zeros(T,n),zeros(T,m))
end
Base.size(H::AuxiliaryLinearization)=(H.nphys+H.naux,H.nphys+H.naux)
Base.eltype(H::AuxiliaryLinearization{TA,TB,TD,T}) where {TA,TB,TD,T}=T

function LinearAlgebra.mul!(y::AbstractVector, H::AuxiliaryLinearization, x::AbstractVector)
    n=H.nphys; N=n+H.naux
    length(x)==N && length(y)==N || throw(DimensionMismatch())
    xp=@view x[1:n]; xa=@view x[n+1:N]; yp=@view y[1:n]; ya=@view y[n+1:N]
    mul!(yp,H.A,xp); mul!(H.work_phys,H.B,xa); yp .+= H.work_phys
    mul!(ya,H.D,xa); mul!(H.work_aux,adjoint(H.B),xp); ya .+= H.work_aux
    y
end
function Base.:*(H::AuxiliaryLinearization,x::AbstractVector)
    y=zeros(promote_type(eltype(H),eltype(x)),size(H,1)); mul!(y,H,x); y
end
lift_rhs(H::AuxiliaryLinearization,b::AbstractVector)=vcat(b,zeros(promote_type(eltype(H),eltype(b)),H.naux))
physical_part(H::AuxiliaryLinearization,x::AbstractVector)=@view x[1:H.nphys]

function schur_matrix(A,B,D,z)
    n=size(A,1)
    A-z*I(n)-B*((D-z*I(size(D,1)))\adjoint(B))
end
