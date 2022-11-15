mutable struct LBreIF{T}
    maxiter::Int            # max number of iterations
    verbose::Bool           # whether to show procedural information
    tol::T                  # change tolerance upon convergence
    rho::Real                  # step size
    mu::Real                   # L1 regularization coefficient

    function LBreIF{T}(;maxiter::Integer = 200,
                        verbose::Bool = false,
                        tol::Real=cbrt(eps(T)),
                        rho::Real = convert(T, 0.5),
                        mu::Real = convert(T,1)) where T
        maxiter > 1 || throw(ArgumentError("maxiter must be greater than 1."))
        tol > 0 || throw(ArgumentError("tol must be positive."))
        rho < 1 || throw(ArgumentError("rho must be smaller than 1."))
        rho > 0 || throw(ArgumentError("rho must be greater than 0."))
        new{T}(maxiter, verbose, tol, rho, mu)
    end
end

function solve!(alg::LBreIF{T}, A, X, Y) where {T}
    nmf_skeleton!(LBreIFUpd{T}(alg.rho, alg.mu), A, X, Y, alg.maxiter, alg.verbose, alg.tol)
end

struct LBreIFUpd{T} <: NMFUpdater{T}
    rho::Real
    mu::Real
end

struct LBreIFUpd_State{T}
    Vx::Array{T}
    Vy::Array{T}
    function LBreIFUpd_State{T}(A, X::Matrix{T}, Y::Matrix{T}) where T
        m, n, r = nmf_checksize(A, X, Y)
        new{T}(Array{T}(undef, m*r), Array{T}(undef, n*r))
    end
end

prepare_state(::LBreIFUpd{T}, A, X, Y) where {T} = LBreIFUpd_State{T}(A, X, Y)
# evaluate_objv(::LBreIFUpd{T}, s::LBreIFUpd_State{T}, A, X, Y) where T = convert(T, 0.5) * sqL2dist(A, s.XY)


function update_xy!(upd::LBreIFUpd{T}, s::LBreIFUpd_State{T}, A, X::Matrix{T}, Y::Matrix{T}, Px::Matrix{T}, Py::Matrix{T}, j_k::Integer) where T
    # fields

    rho = upd.rho
    mu = upd.mu
    Vx = s.Vx
    Vy = s.Vy
    m, n, r = nmf_checksize(A, X, Y)
    x = X[:]
    y = transpose(Y)[:]
    px = Px[:]
    py = transpose(Py)[:]

    # update x_j
    Vx = 1/rho*(norm(x)^2 + norm(y)^2 + 1)*x + mu*px - (kron(I(r), (X*Y - A)))*y
    v = soft_thresholding(rho*Vx, rho*mu)
    f(t) = norm(v)^2*t^3 + t - 1
    t_0 = fzero(f, 0)
    x_1 = t_0*v
    px = px - 1/(rho*mu)*((norm(x_1)^2 + norm(y)^2 + 1)*x_1 - (norm(x)^2 + norm(y)^2 + 1)*x + rho*(kron(I(r), (X*Y - A)))*y)

    X1 = reshape(x_1, m, r)
    XY1 = X1*Y
    # update y_j
    Vy = 1/rho*(norm(x_1)^2 + norm(y)^2 + 1)*y+ mu*py - transpose(kron(I(r), (XY1 - A)))*x_1
    v = soft_thresholding(rho*Vy, rho*mu)
    g(t) = norm(v)^2*t^3 + t - 1
    t_0 = fzero(g, 0)
    y_1 = t_0*v
    py = py - 1/(rho*mu)*((norm(x_1)^2 + norm(y_1)^2 + 1)*y_1 - (norm(x_1)^2 + norm(y)^2 + 1)*y + rho*transpose(kron(I(r), (XY1 - A)))*x_1)

    X = reshape(x_1, m, r)
    Y = copy(transpose(reshape(y_1, n, r)))
    Px = reshape(px, m, r)
    Py = copy(transpose(reshape(py, n, r)))
    return X, Y, Px, Py
end