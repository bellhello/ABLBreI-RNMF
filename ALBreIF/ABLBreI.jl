mutable struct ABLBreI{T}
    runtime::Int
    verbose::Bool           # whether to show procedural information
    τ::T                  # change tolerance upon convergence
    ρ::T                  # step size
    μ₁::T                 # L1 regularization coefficient for X
    μ₂::T 

    function ABLBreI{T}(; runtime::Integer=200,
        verbose::Bool=false,
        τ::Real=∛eps(T),
        ρ::Real=convert(T, 0.5),
        μ₁::Real=convert(T, 0.01),
        μ₂::Real=convert(T, 0.01)) where {T}
        runtime ≥ 1 || throw(ArgumentError("runtime must be greater than 0."))
        τ > 0 || throw(ArgumentError("τ must be positive."))
        0 < ρ < 1 || throw(ArgumentError("ρ must be smaller than 1 and greater than 0."))
        new{T}(runtime, verbose, τ, ρ, μ₁, μ₂)
    end
end

function solve!(alg::ABLBreI{T}, A, X, Y) where {T}
    nmf_skeleton!(ABLBreIUpd{T}(alg.ρ, alg.μ₁, alg.μ₂), A, X, Y, alg.runtime, alg.verbose, alg.τ)
end

struct ABLBreIUpd{T} <: NMFUpdater{T}
    ρ::T
    μ₁::T
    μ₂::T
end

struct ABLBreIUpd_State{T}
    Vx::Array{T}
    Vy::Array{T}
    function ABLBreIUpd_State{T}(A, X::Matrix{T}, Y::Matrix{T}) where {T}
        m, n, r = nmf_checksize(A, X, Y)
        new{T}(Array{T}(undef, m * r), Array{T}(undef, n * r))
    end
end

prepare_state(::ABLBreIUpd{T}, A, X, Y) where {T} = ABLBreIUpd_State{T}(A, X, Y)
# evaluate_objv(::ABLBreIUpd{T}, s::ABLBreIUpd_State{T}, A, X, Y) where T = convert(T, 0.5) * sqL2dist(A, s.XY)

function update_xy!(upd::ABLBreIUpd{T}, s::ABLBreIUpd_State{T}, A, X::Matrix{T}, Y::Matrix{T}, Px::Matrix{T}, Py::Matrix{T}, jₖ) where {T} # jₖ is the index of the j-th column of X and Y
    # fields

    ρ = upd.ρ
    μ₁ = upd.μ₁
    μ₂ = upd.μ₂
    Vx = s.Vx
    Vy = s.Vy
    xⱼ = X[:, jₖ]
    yⱼ = Y[jₖ, :]

    # update x_j
    Vx = 1 / ρ * (dot(xⱼ, xⱼ) + dot(yⱼ, yⱼ) + 1) * xⱼ + Px[:, jₖ] - (X * Y - A) * yⱼ
    projectnn!(Vx)
    v = soft_thresholding(ρ * Vx, ρ * μ₁)
    f(t) = dot(v, v) * t^3 + (dot(yⱼ, yⱼ) + 1) * t - 1
    t_0 = find_positive_root(f, 0, 1, 1e-6)
    v = t_0 * v
    # projectnn!(v)
    x_1 = v
    Px[:, jₖ] = Px[:, jₖ] - 1 / (ρ ) * ((dot(x_1, x_1) + dot(yⱼ, yⱼ) + 1) * x_1 - (dot(xⱼ, xⱼ) + dot(yⱼ, yⱼ) + 1) * xⱼ + ρ * (X * Y - A) * yⱼ)

    # update y_j
    Vy = 1 / ρ * (dot(x_1, x_1) + dot(yⱼ, yⱼ) + 1) * yⱼ + Py[jₖ, :] - ρ * transpose(X * Y - xⱼ * transpose(yⱼ) + x_1 * transpose(yⱼ) - A) * x_1
    projectnn!(Vy)
    v = soft_thresholding(ρ * Vy, ρ * μ₂)
    g(t) = dot(v, v) * t^3 + (dot(x_1, x_1) + 1) * t - 1
    t_0 = find_positive_root(g, 0, 1, 1e-6)
    v = t_0 * v
    # projectnn!(v)
    y_1 = v
    Py[jₖ, :] = Py[jₖ, :] - 1 / (ρ ) * ((dot(x_1, x_1) + dot(y_1, y_1) + 1) * y_1 - (dot(x_1, x_1) + dot(yⱼ, yⱼ) + 1) * yⱼ + transpose(X * Y - X[:, jₖ] * transpose(yⱼ) + x_1 * transpose(yⱼ) - A) * x_1)
    
    # update X and Y
    X[:, jₖ] = x_1
    Y[jₖ, :] = y_1
    return X, Y, Px, Py
end