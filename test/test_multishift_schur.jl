using Test, LinearAlgebra, Random
Random.seed!(42)
@testset "Schur + one-basis multi-shift" begin
 n,m=10,6; A=randn(ComplexF64,n,n); A=A+A'; D=Diagonal(collect(2.0:0.4:4.0)); B=0.2randn(ComplexF64,n,m); b=randn(ComplexF64,n); zs=ComplexF64[-.7+.08im,.1+.08im,.9+.08im,1.7+.08im]
 X,info=schur_multishift(A,B,D,b,zs;krylovdim=n+m,tol=1e-11)
 for (x,z) in zip(X,zs); @test x ≈ (schur_matrix(A,B,D,z)\b) rtol=2e-9 atol=2e-10; end
 @test info.iterations <= n+m
end
