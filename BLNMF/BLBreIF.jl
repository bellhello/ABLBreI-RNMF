mutable struct BLBreIF{T}
    runtime::Int
    verbose::Bool           # whether to show procedural information
    τ::T                  # change tolerance upon convergence
    ρ::T                  # step size
    μ::T                   # L1 regularization coefficient

    function BLBreIF{T}(; runtime::Integer=200,
        verbose::Bool=false,
        τ::Real=∛eps(T),
        ρ::Real=convert(T, 0.5),
        μ::Real=convert(T, 1)) where {T}
        runtime ≥ 1 || throw(ArgumentError("runtime must be greater than 0."))
        τ > 0 || throw(ArgumentError("τ must be positive."))
        0 < ρ < 1 || throw(ArgumentError("ρ must be smaller than 1 and greater than 0."))
        new{T}(runtime, verbose, τ, ρ, μ)
    end
end

function solve!(alg::BLBreIF{T}, A, X, Y) where {T}
    nmf_skeleton!(BLBreIFUpd{T}(alg.ρ, alg.μ), A, X, Y, alg.runtime, alg.verbose, alg.τ)
end

struct BLBreIFUpd{T} <: NMFUpdater{T}
    ρ::T
    μ::T
end

struct BLBreIFUpd_State{T}
    Vx::Array{T}
    Vy::Array{T}
    function BLBreIFUpd_State{T}(A, X::Matrix{T}, Y::Matrix{T}) where {T}
        m, n, r = nmf_checksize(A, X, Y)
        new{T}(Array{T}(undef, m * r), Array{T}(undef, n * r))
    end
end

prepare_state(::BLBreIFUpd{T}, A, X, Y) where {T} = BLBreIFUpd_State{T}(A, X, Y)
# evaluate_objv(::BLBreIFUpd{T}, s::BLBreIFUpd_State{T}, A, X, Y) where T = convert(T, 0.5) * sqL2dist(A, s.XY)

function update_xy!(upd::BLBreIFUpd{T}, s::BLBreIFUpd_State{T}, A, X::Matrix{T}, Y::Matrix{T}, Px::Matrix{T}, Py::Matrix{T}, jₖ) where {T}
    # fields

    ρ = upd.ρ
    μ = upd.μ
    Vx = s.Vx
    Vy = s.Vy

    # update x_j
    Vx = 1 / ρ * (norm(X[:, jₖ])^2 + norm(Y[jₖ, :])^2 + 1) * X[:, jₖ] + μ * Px[:, jₖ] - (X * Y - A) * Y[jₖ, :]
    v = soft_thresholding(ρ * Vx, ρ * μ)
    f(t) = norm(v, 2)^2 * t^3 + t - 1
    t_0 = fzero(f, 0)
    x_1 = t_0 * v
    Px[:, jₖ] = Px[:, jₖ] - 1 / (ρ * μ) * ((norm(x_1)^2 + norm(Y[jₖ, :])^2 + 1) * x_1 - (norm(X[:, jₖ])^2 + norm(Y[jₖ, :])^2 + 1) * X[:, jₖ] + ρ * (X * Y - A) * Y[jₖ, :])

    # update y_j
    Vy = 1 / ρ * (norm(x_1)^2 + norm(Y[jₖ, :])^2 + 1) * Y[jₖ, :] + μ * Py[jₖ, :] - ρ * transpose(X * Y - X[:, jₖ] * transpose(Y[jₖ, :]) + x_1 * transpose(Y[jₖ, :]) - A) * x_1
    v = soft_thresholding(ρ * Vy, ρ * μ)
    g(t) = norm(v)^2 * t^3 + t - 1
    t_0 = fzero(g, 0)
    y_1 = t_0 * v
    Py[jₖ, :] = Py[jₖ, :] - 1 / (ρ * μ) * ((norm(x_1)^2 + norm(y_1)^2 + 1) * y_1 - (norm(x_1)^2 + norm(Y[jₖ, :])^2 + 1) * Y[jₖ, :] + transpose(X * Y - X[:, jₖ] * transpose(Y[jₖ, :]) + x_1 * transpose(Y[jₖ, :]) - A) * x_1)

    X[:, jₖ] = x_1
    Y[jₖ, :] = y_1
    return X, Y, Px, Py
end