"""
    auxiliary_matrix(M)

Frequency-independent enlarged matrix [A B; B† D]; (auxiliary_matrix(M) - zI)
is the full (unreduced) enlarged system whose Schur complement is S(z).
"""
auxiliary_matrix(M::SchurModel) = [Matrix(M.A) Matrix(M.B); Matrix(M.B') Matrix(M.D)]

"""
    solve_full_upper(M, z, b)

Upper block x of the solution of
`[A-zI B; B† D-zI] [x; y] = [b; 0]`.
"""
function solve_full_upper(M::SchurModel, z::Number, b)
    n = size(M.A, 1)
    H = auxiliary_matrix(M)
    rhs = [b; zeros(eltype(H), size(M.D, 1), size(b)[2:end]...)]
    sol = (H - z * I) \ rhs
    sol[1:n, ntuple(_ -> :, ndims(b) - 1)...]
end
