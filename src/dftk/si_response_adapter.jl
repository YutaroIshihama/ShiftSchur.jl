module ShiftSchurDFTKSiResponse
using LinearAlgebra
using ShiftSchur
export DFTKResponseNotImplemented, SiResponseProblem, build_si_response_problem, validate_response_problem
struct DFTKResponseNotImplemented <: Exception
    msg::String
end
Base.showerror(io::IO,e::DFTKResponseNotImplemented)=print(io,e.msg)
struct SiResponseProblem{TM,TV}
    model::TM
    probe::TV
    metadata::NamedTuple
end
function validate_response_problem(p::SiResponseProblem)
    M=p.model; b=p.probe
    n=size(M.A,1); m=size(M.D,1)
    size(M.A)==(n,n) || throw(DimensionMismatch("A must be square"))
    size(M.D)==(m,m) || throw(DimensionMismatch("D must be square"))
    size(M.B)==(n,m) || throw(DimensionMismatch("B must be n x m"))
    length(b)==n || throw(DimensionMismatch("probe length must equal size(A,1)"))
    ishermitian(Matrix(M.A)) || error("A is not Hermitian")
    ishermitian(Matrix(M.D)) || error("D is not Hermitian")
    return true
end
function build_si_response_problem(payload; q=(0.0,0.0,0.0), perturbation=:density_head)
    throw(DFTKResponseNotImplemented("The real DFTK silicon SCF succeeded, but the physical response adapter is not connected yet. Implement this function from the installed DFTK linear-response/Sternheimer API. It must construct physical A, B, D and probe b; do not substitute a KS occupied/virtual block partition as a surrogate."))
end
end
