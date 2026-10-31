using Test, LinearAlgebra, ShiftSchur
include(joinpath(@__DIR__,"..","src","dftk","si_response_adapter.jl"))
using .ShiftSchurDFTKSiResponse
@testset "DFTK Si response adapter structural validation" begin
 A=Hermitian([1.0 0.1;0.1 2.0])|>Matrix; D=Hermitian([3.0 0.2;0.2 4.0])|>Matrix; B=[0.2 0.1;0.3 0.2]
 p=SiResponseProblem(SchurModel(A,D,B),ComplexF64[1,0],(source=:unit_test,))
 @test validate_response_problem(p)
end
