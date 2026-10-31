"""
error_estimators.jl — a posteriori estimators for Σ(z)b

Cances論文 Section 6 の adaptive extra bands を動的版に拡張する際に
exact self_energy を参照しないためのモジュール。

- residual_estimator: ||r_k(z)|| ベース、理論保証あり
- two_space_estimator: ||Σ_{k+1}-Σ_k|| ベース、実装最速
"""
module ErrorEstimators

using LinearAlgebra

export residual_estimator, two_space_estimator, residual_norms_grid

"""
DQ cache: D*Q を1回計算して再利用 (m*k 行列)
"""
function compute_DQ(D::AbstractMatrix, Q::AbstractMatrix)
    return D * Q
end

"""
residual_estimator(RM, DQ, z, b)

r_k(z) = (D - zI) Q y_k - B'b
y_k = (T - zI)^{-1} Bred' b
Returns (est_norm, r_vec, y_k)
est_norm = ||r_k|| / dist(z) * ||B||  ~ ||Σ-Σ_k||
"""
function residual_estimator(RM, DQ::AbstractMatrix, z::Number, b::AbstractVector, B::AbstractMatrix)
    # B' b -> actually B' is starting block W, but for Σ*b we need B' * (?) 
    # ここでは B' b の代わりに W = B' を想定した汎化: W = B' (m x n)
    # b は n 次元、Wb = B' * b? ではない、Σ*b = B (D-zI)^{-1} B' b なので
    # Wb = B' * b (mベクトル)
    T = RM.T
    Bred = RM.Bred # n x k
    Q = RM.Q       # m x k

    # y_k = (T - zI)^{-1} (Bred' b)
    rhs_k = Bred' * b
    yk = (Matrix(T) - z*I) \ rhs_k

    # r_k = (D*Q - z Q) yk - B' b
    # B' b = ?
    # B' は m x n ではなく n x m の転置。実際 B (n x m) なので
    # なので B' * b は m ベクトル
    # ここでは B を引数で渡す
    Bp_b = B' * b  # m x 1
    rk = DQ * yk - z * (Q * yk) - Bp_b
    # スケーリング: ||B|| / dist(z, spec(D)) 近似
    # dist は |Im(z)| で下から抑える (η)
    dist_est = max(abs(imag(z)), 1e-12)
    Bnorm = opnorm(B)
    est = Bnorm * norm(rk) / dist_est
    # 相対化用分母: ||Σ_k b||
    Σk_b = Bred * yk
    denom = norm(Σk_b)
    rel_est = denom==0 ? norm(rk) : est / max(denom,1e-16)
    return (abs_est=est, rel_est=rel_est, residual=rk, yk=yk)
end

"""
residual_norms_grid(RM, DQ, ωgrid, η, b, B) -> Vector{Float64}

ωgrid全体の residual estimator を一括計算。greedy選択用。
O(k^3 + m*k*n_ω) だが m*k は DQ キャッシュで1回。
"""
function residual_norms_grid(RM, DQ, ωgrid, η::Real, b, B)
    [ residual_estimator(RM, DQ, ComplexF64(ω+im*η), b, B).rel_est for ω in ωgrid ]
end

"""
two_space_estimator(RM_k, RM_k1, z, b)

||Σ_{k+1} b - Σ_k b|| を誤差とみなす。
RM_k1 は RM_k に1ブロック追加したもの。
コストは k^3 のみ、m に依存しない。
"""
function two_space_estimator(RM_k, RM_k1, z::Number, b)
    Σk_b  = RM_k.Bred * ((Matrix(RM_k.T) - z*I) \ (RM_k.Bred' * b))
    Σk1_b = RM_k1.Bred * ((Matrix(RM_k1.T) - z*I) \ (RM_k1.Bred' * b))
    return norm(Σk1_b - Σk_b) / max(norm(Σk1_b),1e-16)
end

"""
expand_DQ(D, Q_old, DQ_old, Q_new)

Incrementally extend a cached D*Q when Q_new appends columns to Q_old.
"""
function expand_DQ(D, Q_old, DQ_old, Q_new)
    if size(Q_new,2) == size(Q_old,2)
        return DQ_old
    end
    k_old = size(Q_old,2)
    Q_add = Q_new[:, k_old+1:end]
    DQ_add = D * Q_add
    return hcat(DQ_old, DQ_add)
end

end # module
