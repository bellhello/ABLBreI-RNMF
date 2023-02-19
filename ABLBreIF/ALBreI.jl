mutable struct ALBreI{T}
    runtime::Int
    verbose::Bool           # whether to show procedural information
    τ::T                  # change tolerance upon convergence
    ρ::Real                  # step size
    μ₁::Real                   # L1 regularization coefficient
    μ₂::Real

    function ALBreI{T}(; runtime::Integer=200,
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

function solve!(alg::ALBreI{T}, A, X, Y) where {T}
    nmf_skeleton!(ALBreIUpd{T}(alg.ρ, alg.μ₁, alg.μ₂), A, X, Y, alg.runtime, alg.verbose, alg.τ)
end

struct ALBreIUpd{T} <: NMFUpdater{T}
    ρ::Real
    μ₁::Real
    μ₂::Real
end

struct ALBreIUpd_State{T}
    Vx::Array{T}
    Vy::Array{T}
    function ALBreIUpd_State{T}(A, X::Matrix{T}, Y::Matrix{T}) where {T}
        m, n, r = nmf_checksize(A, X, Y)
        new{T}(Array{T}(undef, m * r), Array{T}(undef, n * r))
    end
end

prepare_state(::ALBreIUpd{T}, A, X, Y) where {T} = ALBreIUpd_State{T}(A, X, Y)
# evaluate_objv(::ALBreIUpd{T}, s::ALBreIUpd_State{T}, A, X, Y) where T = convert(T, 0.5) * sqL2dist(A, s.XY)


function update_xy!(upd::ALBreIUpd{T}, s::ALBreIUpd_State{T}, A, X::Matrix{T}, Y::Matrix{T}, Px::Matrix{T}, Py::Matrix{T}, jₖ::Integer) where {T}
    # fields

    ρ = upd.ρ
    μ₁ = upd.μ₁
    μ₂ = upd.μ₂
    Vx = s.Vx
    Vy = s.Vy
    m, n, r = nmf_checksize(A, X, Y)
    x = X[:]
    y = transpose(Y)[:]
    px = Px[:]
    py = transpose(Py)[:]

    # update x_j
    Vx = 1 / ρ * (norm(x)^2 + norm(y)^2 + 1) * x + μ₁ * px - (kron(I(r), (X * Y - A))) * y
    v = soft_thresholding(ρ * Vx, ρ * μ₁)
    f(t) = norm(v)^2 * t^3 + (norm(y)^2 + 1) * t - 1
    t_0 = fzero(f, 0)
    x_1 = t_0 * v
    px = px - 1 / (ρ * μ₁) * ((norm(x_1)^2 + norm(y)^2 + 1) * x_1 - (norm(x)^2 + norm(y)^2 + 1) * x + ρ * (kron(I(r), (X * Y - A))) * y)

    X1 = reshape(x_1, m, r)
    XY1 = X1 * Y
    # update y_j
    Vy = 1 / ρ * (norm(x_1)^2 + norm(y)^2 + 1) * y + μ₂ * py - transpose(kron(I(r), (XY1 - A))) * x_1
    v = soft_thresholding(ρ * Vy, ρ * μ₂)
    g(t) = norm(v)^2 * t^3 + (norm(x_1)^2 + 1) * t - 1
    t_0 = fzero(g, 0)
    y_1 = t_0 * v
    py = py - 1 / (ρ * μ₂) * ((norm(x_1)^2 + norm(y_1)^2 + 1) * y_1 - (norm(x_1)^2 + norm(y)^2 + 1) * y + ρ * transpose(kron(I(r), (XY1 - A))) * x_1)

    X = reshape(x_1, m, r)
    Y = copy(transpose(reshape(y_1, n, r)))
    Px = reshape(px, m, r)
    Py = copy(transpose(reshape(py, n, r)))
    return X, Y, Px, Py
end