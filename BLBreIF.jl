mutable struct BLBreIF{T}
    runtime::Int            
    verbose::Bool           # whether to show procedural information
    tol::T                  # change tolerance upon convergence
    rho::T                  # step size
    mu::T                   # L1 regularization coefficient

    function BLBreIF{T}(;runtime::Integer = 200,
                        verbose::Bool = false,
                        tol::Real=cbrt(eps(T)),
                        rho::Real = convert(T, 0.5),
                        mu::Real = convert(T,1)) where T
        runtime > 1 || throw(ArgumentError("runtime must be greater than 1."))
        tol > 0 || throw(ArgumentError("tol must be positive."))
        rho < 1 || throw(ArgumentError("rho must be smaller than 1."))
        rho > 0 || throw(ArgumentError("rho must be greater than 0."))
        new{T}(runtime, verbose, tol, rho, mu)
    end
end

function solve!(alg::BLBreIF{T}, A, X, Y) where {T}
    nmf_skeleton!(BLBreIFUpd{T}(alg.rho, alg.mu), A, X, Y, alg.runtime, alg.verbose, alg.tol)
end

struct BLBreIFUpd{T} <: NMFUpdater{T}
    rho::T
    mu::T
end

struct BLBreIFUpd_State{T}
    Vx::Array{T}
    Vy::Array{T}
    function BLBreIFUpd_State{T}(A, X::Matrix{T}, Y::Matrix{T}) where T
        m, n, r = nmf_checksize(A, X, Y)
        new{T}(Array{T}(undef, m*r), Array{T}(undef, n*r))
    end
end

prepare_state(::BLBreIFUpd{T}, A, X, Y) where {T} = BLBreIFUpd_State{T}(A, X, Y)
# evaluate_objv(::BLBreIFUpd{T}, s::BLBreIFUpd_State{T}, A, X, Y) where T = convert(T, 0.5) * sqL2dist(A, s.XY)

function update_xy!(upd::BLBreIFUpd{T}, s::BLBreIFUpd_State{T}, A, X::Matrix{T}, Y::Matrix{T}, Px::Matrix{T}, Py::Matrix{T}, j_k) where T
    # fields

    rho = upd.rho
    mu = upd.mu
    Vx = s.Vx
    Vy = s.Vy
    
    # update x_j
    Vx = 1/rho*(norm(X[:, j_k])^2 + norm(Y[j_k, :])^2 + 1)*X[:, j_k] + mu*Px[:, j_k] - (X*Y - A)*Y[j_k, :]
    v = soft_thresholding(rho*Vx, rho*mu)
    f(t) = norm(v, 2)^2*t^3 + t - 1
    t_0 = fzero(f, 0)
    x_1 = t_0*v
    Px[:, j_k] = Px[:, j_k] - 1/(rho*mu)*((norm(x_1)^2 + norm(Y[j_k, :])^2 + 1)*x_1 - (norm(X[:, j_k])^2 + norm(Y[j_k, :])^2 + 1)*X[:, j_k] + rho*(X*Y - A)*Y[j_k, :])

    # update y_j
    Vy = 1/rho*(norm(x_1)^2 + norm(Y[j_k, :])^2 + 1)*Y[j_k, :]+ mu*Py[j_k, :] - rho*transpose(X*Y - X[:, j_k]*transpose(Y[j_k, :]) + x_1*transpose(Y[j_k, :]) - A)*x_1
    v = soft_thresholding(rho*Vy, rho*mu)
    g(t) = norm(v)^2*t^3 + t - 1
    t_0 = fzero(g, 0)
    y_1 = t_0*v
    Py[j_k, :] = Py[j_k, :] - 1/(rho*mu)*((norm(x_1)^2 + norm(y_1)^2 + 1)*y_1 - (norm(x_1)^2 + norm(Y[j_k, :])^2 + 1)*Y[j_k, :] + transpose(X*Y - X[:, j_k]*transpose(Y[j_k, :]) + x_1*transpose(Y[j_k, :]) - A)*x_1)

    X[:, j_k] = x_1
    Y[j_k, :] = y_1
    return X, Y, Px, Py
end