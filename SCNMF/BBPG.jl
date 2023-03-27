mutable struct BBPG{T}
    obj::Symbol
    constraint::Real
    runtime::Int
    ρ::Real

    function BBPG{T}(; obj::Symbol=:stan,
        constraint::Real=1,
        runtime::Integer=30,
        ρ::Real=0.1) where {T}
        obj == :stan || obj == :cons || throw(ArgumentError("Invalid value for obj."))
        constraint > 0 && constraint <= 1 || throw(ArgumentError("constraint must be greater than 0"))
        runtime ≥ 1 || throw(ArgumentError("runtime must be greater than 0."))
        ρ > 0 && ρ < 1 || throw(ArgumentError("ρ must be greater than 0 and smaller than 1."))
        new{T}(obj, constraint, runtime, ρ)
    end
end

function solve!(alg::BBPG{T}, X, W, H) where {T}
    if alg.obj == :stan
        nmf_skeleton!(BBPGUpdSTAN(sqrt(eps(T)), alg.ρ), X, W, H, alg.runtime)
    else
        nmf_skeleton!(BBPGUpdCONS(sqrt(eps(T)), alg.constraint, alg.ρ), X, W, H, alg.runtime)
    end
end

struct BBPGUpdSTAN{T} <: NMFUpdater{T}
    δ::T
    ρ::Real
end

struct BBPGUpdSTAN_State{T}
    vx::Array{T}
    vy::Array{T}

    function BBPGUpdSTAN_State{T}(X, W::Matrix{T}, H::Matrix{T}) where {T}
        p, n, r = nmf_checksize(X, W, H)
        new{T}(Array{T}(undef, p), Array{T}(undef, n))
    end
end

prepare_state(::BBPGUpdSTAN{T}, X, W, H) where {T} = BBPGUpdSTAN_State{T}(X, W, H)
# evaluate_objv(::BBPGUpdSTAN{T}, s::BBPGUpdSTAN_State{T}, X, W, H) where T = sqrt(sqL2dist(X, s.WH))

function update_wh!(upd::BBPGUpdSTAN{T}, s::BBPGUpdSTAN_State{T}, X, W::Matrix{T}, H::Matrix{T}, j::Integer) where {T}

    #p = size(X, 1)
    #n = size(X, 2)
    #r = size(W, 2)

    xⱼ = W[:, j]
    yⱼ = transpose(H)[:, j]
    vx = s.vx
    vy = s.vy
    ρ = upd.ρ
    δ = upd.δ

    μₖ = norm(W * H - X - xⱼ * transpose(yⱼ), 2)

    #compute vx
    vx = 1 / ρ * (norm(xⱼ, 2)^2 + norm(yⱼ, 2)^2 + μₖ) * xⱼ - (W * H - X) * yⱼ

    #compute vy
    vy = 1 / ρ * (norm(xⱼ, 2)^2 + norm(yⱼ, 2)^2 + μₖ) * yⱼ - transpose(W * H - X) * xⱼ

    #project vx and vy
    projectnn!(vx)
    projectnn!(vy)

    #compute theta
    m(theta) = (norm(vx, 2)^2 + norm(vy, 2)^2) * theta^3 + μₖ * theta - ρ
    th = fzero(m, δ, 10^4)

    #compute x^(k+1) and y^(k+1)
    xⱼ = th * vx
    yⱼ = th * vy

    #update W and H
    W[:, j] = xⱼ
    H[j, :] = yⱼ
    return W, H
end

struct BBPGUpdCONS{T} <: NMFUpdater{T}
    δ::T
    constraint::Real
    ρ::Real
end

struct BBPGUpdCONS_State{T}
    vx::Array{T}
    vy::Array{T}

    function BBPGUpdCONS_State{T}(X, W::Matrix{T}, H::Matrix{T}) where {T}
        p, n, r = nmf_checksize(X, W, H)
        new{T}(Array{T}(undef, p), Array{T}(undef, n))
    end
end

prepare_state(::BBPGUpdCONS{T}, X, W, H) where {T} = BBPGUpdCONS_State{T}(X, W, H)
# evaluate_objv(::BBPGUpdCONS{T}, s::BBPGUpdCONS_State{T}, X, W, H) where T = sqrt(sqL2dist(X, s.WH))

function update_wh!(upd::BBPGUpdCONS{T}, s::BBPGUpdCONS_State{T}, X, W::Matrix{T}, H::Matrix{T}, j::Integer) where {T}

    #p = size(X, 1)
    #n = size(X, 2)
    #r = size(W, 2)

    xⱼ = W[:, j]
    yⱼ = transpose(H)[:, j]
    vx = s.vx
    vy = s.vy
    ρ = upd.ρ
    δ = upd.δ
    constraint = 1 - upd.constraint

    μₖ = norm(W * H - X - xⱼ * transpose(yⱼ), 2)

    #compute vx
    vx = 1 / ρ * (norm(xⱼ, 2)^2 + norm(yⱼ, 2)^2 + μₖ) * xⱼ - (W * H - X) * yⱼ

    #compute vy
    vy = 1 / ρ * (norm(xⱼ, 2)^2 + norm(yⱼ, 2)^2 + μₖ) * yⱼ - transpose(W * H - X) * xⱼ


    #project vx and vy
    projectnn!(vx)
    projectnn!(vy)

    #hard_thresholding
    hard_thresholding(vx, constraint)
    hard_thresholding(vy, constraint)

    #compute theta
    m(theta) = (norm(vx, 2)^2 + norm(vy, 2)^2) * theta^3 + μₖ * theta - ρ
    th = fzero(m, δ, 10^4)

    #compute x^(k+1) and y^(k+1)
    xⱼ = th * vx
    yⱼ = th * vy

    W[:, j] = xⱼ
    H[j, :] = yⱼ
    return W, H
end