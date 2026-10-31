using Test
# Second Si regression: keep this separate from the existing spectrum/epsilon regression.
# The purpose is solver equivalence on the SAME silicon response problem, not a second physics target.
# Existing repository should provide `silicon_response_fixture()` returning (A,B,D,b,zs).
@testset "Si: Schur direct vs auxiliary multi-shift" begin
 if isdefined(@__MODULE__, :silicon_response_fixture)
   A,B,D,b,zs=silicon_response_fixture()
   X,info=schur_multishift(A,B,D,b,zs;krylovdim=min(size(A,1)+size(D,1),120),tol=1e-8)
   for (x,z) in zip(X,zs)
     xref=schur_matrix(A,B,D,z)\b
     @test norm(x-xref)/max(norm(xref),eps()) < 2e-7
   end
 else
   @test_skip "Wire silicon_response_fixture() to the existing Si fixture"
 end
end
