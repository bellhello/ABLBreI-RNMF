mutable struct BLBreIF{T}
    maxiter::Int            # max number of iterations
    verbose::Bool           # whether to show procedural information
    rho::T                  # step size
    mu::T                   # L1 regularization coefficient

    function BLBreIF{T}(;maxiter::Integer = 200,
                        verbose::Bool = false,
                        rho::Real = convert(T, 0.5),
                        mu::Real = convert(T,1)) where T
        maxiter > 1 || throw(ArgumentError("maxiter must be greater than 1."))
        rho < 1 || throw(ArgumentError("rho must be smaller than 1."))
        rho > 0 || throw(ArgumentError("rho must be greater than 0."))
        new{T}(maxiter, verbose, rho, mu)
    end
end

function solve!(alg::BLBreIF{T}, A, X, Y) where {T}
    nmf_skeleton!(BLBreIFUpd{T}(sqrt(eps(T)), alg::rho, alg::mu), A, X, Y, alg.maxiter, alg.verbose)
end

struct BLBreIFUpd{T} <: NMFUpdater{T}
    delta::T
    rho::T
    mu::T
end

struct BLBreIFUpd_State{T}
    XY::Matrix{T}
    Vx::Matrix{T}
    Vy::Matrix{T}
    px::Matrix{T}
    py::Matrix{T}
    function BLBreIFUpd_State{T}(A, X::Matrix{T}, Y::Matrix{T}) where T
        m, n, r = nmf_checksize(A, X, Y)
        new{T}(X * Y,
            Matrix{T}(undef, m, r),
            Matrix{T}(undef, r, n),
            Matrix{T}(undef, m, r),
            Matrix{T}(undef, r, n))
    end
end

prepare_state(::BLBreIFUpd{T}, A, X, Y) where {T} = BLBreIFUpd_State{T}(A, X, Y)
evaluate_objv(::BLBreIFUpd{T}, s::BLBreIFUpd_State{T}, A, X, Y) where T = convert(T, 0.5) * sqL2dist(A, s.XY)

function update_xy!(upd::BLBreIFUpd{T}, s::BLBreIFUpd_State{T}, A, X:Matrix{T}, Y::Matrix{T}, j_k) where T
    # fields
    m = size(A, 1)
    n = size(A, 2)
    r = size(X, 2)
    rho = upd.rho
    mu = upd.mu
    delta = upd.delta
    XY = s.XY
    Vx = s.Vx
    Vy = s.Vy
    px = s.px
    py = s.py

    # update x_j
    Vx[:, j_k] = 1/rho*(norm(X[:, j_k], 2)^2 + norm(Y[j_k, :], 2)^2 + 1)^X[:, j_k] + mu*px[:, j_k] - (XY - A)*Y[j_k, :]
    
    px[:, j_k] = px[:, j_k] - 1/rho*((norm(x, 2)^2+norm(Y[j_k, :],2)^2+1)*x - (norm(X[:, j_k], 2)^2 + norm(Y[j_k, :], 2)^2 + 1)*X[:, j_k] + rho*(XY - A)*Y[j_k, :])
