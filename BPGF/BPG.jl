mutable struct BPG{T}
    runtime::Int
    verbose::Bool           # whether to show procedural information
    τ::T                  # change tolerance upon convergence
    ρ::Real                  # step size
    μ₁::Real                   # L1 regularization coefficient
    μ₂::Real

    function BPG{T}(; runtime::Integer=200,
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

function solve!(alg::BPG{T}, A, X, Y) where {T}
    nmf_skeleton!(BPGUpd{T}(alg.ρ, alg.μ₁, alg.μ₂), A, X, Y, alg.runtime, alg.verbose, alg.τ)
end

struct BPGUpd{T} <: NMFUpdater{T}
    ρ::Real
    μ₁::Real
    μ₂::Real
end

struct BPGUpd_State{T}
    Vx::Array{T}
    Vy::Array{T}
    function BPGUpd_State{T}(A, X::Matrix{T}, Y::Matrix{T}) where {T}
        m, n, r = nmf_checksize(A, X, Y)
        new{T}(Array{T}(undef, m * r), Array{T}(undef, n * r))
    end
end

prepare_state(::BPGUpd{T}, A, X, Y) where {T} = BPGUpd_State{T}(A, X, Y)
# evaluate_objv(::BPGUpd{T}, s::BPGUpd_State{T}, A, X, Y) where T = convert(T, 0.5) * sqrt(sqL2dist(A, XY))

function update_xy!(upd::BPGUpd{T}, s::BPGUpd_State{T}, A, X::Matrix{T}, Y::Matrix{T}, jₖ::Integer) where {T}

    ρ = upd.ρ
    μ₁ = upd.μ₁
    μ₂ = upd.μ₂
    Vx = s.Vx
    Vy = s.Vy
    m, n, r = nmf_checksize(A, X, Y)
    x = X[:]
    y = transpose(Y)[:]

    #compute Vx
    Vx = 1 / ρ * (dot(x, x) + dot(y, y) + 1) * x - (kron(I(r), (X * Y - A))) * y

    #compute Vy
    Vy = 1 / ρ * (dot(x, x) + dot(y, y) + 1) * y - transpose(kron(I(r), (X * Y - A))) * x

    #project Vx and Vy
    projectnn!(Vx)
    projectnn!(Vy)

    #soft_thresholding
    vx = soft_thresholding(ρ * Vx, ρ * μ₁)
    vy = soft_thresholding(ρ * Vy, ρ * μ₂)
    
    #compute theta
    g(theta) = (dot(vx, vx) + dot(vy, vy)) * theta^3 + theta - 1
    th = find_positive_root(g, 0, 1, 1e-6)

    #compute x^(k+1) and y^(k+1)
    x = th * vx
    y = th * vy

    #update W and H
    X = copy(reshape(x, m, r))
    Y = copy(transpose(reshape(y, n, r)))
    return X, Y
end