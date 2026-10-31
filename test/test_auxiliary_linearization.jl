using Test, LinearAlgebra, Random
Random.seed!(20261031)
@testset "auxiliary linearization equals Schur complement" begin
 n,m=7,4; Q=randn(ComplexF64,n,n); A=Q+Q'; R=randn(ComplexF64,m,m); D=R+R'; B=randn(ComplexF64,n,m); b=randn(ComplexF64,n); z=0.37+0.11im
 H=AuxiliaryLinearization(A,B,D); xa=(Matrix([A B; B' D])-z*I(n+m))\lift_rhs(H,b); xs=schur_matrix(A,B,D,z)\b
 @test physical_part(H,xa) ≈ xs rtol=2e-11 atol=2e-12
end
