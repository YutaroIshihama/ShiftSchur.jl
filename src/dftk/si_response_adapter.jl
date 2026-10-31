module ShiftSchurDFTKSiResponse
using LinearAlgebra
using ShiftSchur
using DFTK

export SiResponseProblem, build_si_response_problem, validate_response_problem
export dftk_static_response_smoke

struct SiResponseProblem{TM,TV}
    model::TM
    probe::TV
    metadata::NamedTuple
end

function validate_response_problem(p::SiResponseProblem)
    M, b = p.model, p.probe
    n, m = size(M.A,1), size(M.D,1)
    size(M.A)==(n,n) || throw(DimensionMismatch("A must be square"))
    size(M.D)==(m,m) || throw(DimensionMismatch("D must be square"))
    size(M.B)==(n,m) || throw(DimensionMismatch("B must be n x m"))
    length(b)==n || throw(DimensionMismatch("probe length must equal size(A,1)"))
    ishermitian(Matrix(M.A)) || error("A is not Hermitian")
    ishermitian(Matrix(M.D)) || error("D is not Hermitian")
    true
end

"""Run a genuine DFTK static independent/interacting-response smoke test.
This validates that the SCF payload is response-capable; it is NOT used to invent
A/B/D blocks.  DFTK's apply_chi0 / solve_Omega+K path is kept as the physical oracle.
"""
function dftk_static_response_smoke(payload; amplitude=1.0)
    scfres = payload.raw
    basis = payload.basis
    # Periodic q=0 test potential, defined on the actual FFT grid.  This avoids
    # the non-periodic position operator while exercising DFTK's true chi0 path.
    rv = DFTK.r_vectors(basis)
    dV = amplitude .* cos.(2pi .* map(r -> r[1], rv))
    dV = cat(dV; dims=4)
    chi0 = DFTK.apply_χ0(scfres, dV)
    return (deltaV=dV, delta_rho0=chi0.δρ, raw=chi0)
end

# Build a finite, physical response sample in density space and obtain a Hermitian
# response surrogate by querying DFTK.apply_chi0 itself. This is a correctness bridge
# for the tiny Si smoke test. It deliberately does NOT use a KS occ/virt block split.
function _sample_static_chi0(payload; nmodes::Int=6)
    scfres, basis = payload.raw, payload.basis
    rv = DFTK.r_vectors(basis)
    shape = size(payload.rho)
    modes = Array{Float64,4}[]
    coords = (1,2,3)
    harmonics = (1,2)
    for h in harmonics, a in coords
        v = cos.(2pi*h .* map(r -> r[a], rv)); push!(modes, cat(v; dims=4))
        length(modes) >= nmodes && break
    end
    modes = modes[1:min(nmodes,length(modes))]
    # Gram matrix in real-space L2 metric and orthonormalise the probes.
    dvol = basis.dvol
    G = [real(sum(conj.(u).*v)*dvol) for u in modes, v in modes]
    F = eigen(Hermitian(G)); keep = findall(>(1e-10*maximum(F.values)), F.values)
    C = F.vectors[:,keep] * Diagonal(1 ./ sqrt.(F.values[keep]))
    probes = [sum(C[j,k].*modes[j] for j in eachindex(modes)) for k in axes(C,2)]
    r = length(probes)
    X = zeros(Float64,r,r)
    responses = Any[]
    for j in 1:r
        out = DFTK.apply_χ0(scfres, probes[j]); push!(responses,out)
        for i in 1:r
            X[i,j] = real(sum(conj.(probes[i]).*out.δρ)*dvol)
        end
    end
    X = Matrix(Hermitian((X+X')/2))
    return X, probes, responses
end

"""Construct the first DFTK-backed ShiftSchur smoke problem.

The finite matrix is obtained by sampling DFTK's *actual* static independent-particle
polarizability on periodic density probes.  We then diagonalize this sampled Hermitian
response and encode it as an exact pole model Sigma(0) = B*D^-1*B'.

This makes the current static Si example executable without pretending that KS
occupied/virtual blocks are the Cances Schur blocks.  `metadata.static_only=true` is
intentional: dynamic frequencies require a frequency-dependent DFTK Sternheimer
oracle and must not be interpreted physically from this static fit.
"""
function build_si_response_problem(payload; q=(0.0,0.0,0.0), perturbation=:density_head,
                                   nmodes::Int=6, pole_scale::Real=1.0)
    q == (0.0,0.0,0.0) || throw(ArgumentError("current DFTK smoke adapter supports q=0 only"))
    X, probes, responses = _sample_static_chi0(payload; nmodes)
    n = size(X,1)
    # chi0 is negative semidefinite in the usual static convention.  Fit -chi0 >= 0.
    F = eigen(Hermitian(-X)); lam = max.(F.values,0.0)
    D = Matrix{ComplexF64}(I,n,n) .* pole_scale
    B = ComplexF64.(F.vectors * Diagonal(sqrt.(lam .* pole_scale)))
    A = zeros(ComplexF64,n,n)
    # Then Sigma(0)=B D^-1 B' ~= -X exactly up to clipped numerical eigenvalues.
    M = SchurModel(A,D,B)
    b = zeros(ComplexF64,n); b[1]=1
    fit = B*(D\B')
    fit_error = norm(fit + X)/max(norm(X),eps())
    metadata = (source=:DFTK_apply_chi0_static, q=q, perturbation=perturbation,
                static_only=true, nmodes=n, chi0_matrix=X, probes=probes,
                raw_responses=responses, static_fit_relative_error=fit_error,
                warning="Static response fit only; do not interpret z != 0 as a physical DFTK dynamic response.")
    p = SiResponseProblem(M,b,metadata); validate_response_problem(p); p
end
end
