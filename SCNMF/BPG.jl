mutable struct BPG{T}
    obj::Symbol
    constraint::Real
    runtime::Int
    ρ::Real

    function BPG{T}(; obj::Symbol=:stan,
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

function solve!(alg::BPG{T}, X, W, H) where {T}
    if alg.obj == :stan
        nmf_skeleton!(BPGUpdSTAN(√eps(T), alg.ρ), X, W, H, alg.runtime)
    else
        nmf_skeleton!(BPGUpdCONS(√eps(T), alg.constraint, alg.ρ), X, W, H, alg.runtime)
    end
end

struct BPGUpdSTAN{T} <: NMFUpdater{T}
    δ::T
    ρ::Real
end

struct BPGUpdSTAN_State{T}
    vx::Array{T}
    vy::Array{T}

    function BPGUpdSTAN_State{T}(X, W::Matrix{T}, H::Matrix{T}) where {T}
        p, n, r = nmf_checksize(X, W, H)
        new{T}(Array{T}(undef, p * r), Array{T}(undef, n * r))
    end
end

prepare_state(::BPGUpdSTAN{T}, X, W, H) where {T} = BPGUpdSTAN_State{T}(X, W, H)
# evaluate_objv(::BPGUpdSTAN{T}, s::BPGUpdSTAN_State{T}, X, W, H) where T = sqrt(sqL2dist(X, s.WH))

function update_wh!(upd::BPGUpdSTAN{T}, s::BPGUpdSTAN_State{T}, X, W::Matrix{T}, H::Matrix{T}, j::Integer) where {T}

    p, n, r = nmf_checksize(X, W, H)
    x = W[:]
    y = transpose(H)[:]
    vx = s.vx
    vy = s.vy
    ρ = upd.ρ
    δ = upd.δ

    #compute vx
    vx = 1 / ρ * (norm(x)^2 + norm(y)^2 + 1) * x - (kron(I(r), (W * H - X))) * y

    #compute vy
    vy = 1 / ρ * (norm(x)^2 + norm(y)^2 + 1) * y - transpose(kron(I(r), (W * H - X))) * x

    #project vx and vy
    projectnn!(vx)
    projectnn!(vy)

    #compute theta
    m(theta) = (norm(vx)^2 + norm(vy)^2) * theta^3 + theta - ρ
    th = fzero(m, δ, 10^4)

    #compute x^(k+1) and y^(k+1)
    x = th * vx
    y = th * vy

    #update W and H
    W = copy(reshape(x, p, r))
    Y = reshape(y, n, r)
    H = copy(transpose(Y))
    return W, H
end

struct BPGUpdCONS{T} <: NMFUpdater{T}
    δ::T
    constraint::Real
    ρ::Real
end

struct BPGUpdCONS_State{T}
    vx::Array{T}
    vy::Array{T}

    function BPGUpdCONS_State{T}(X, W::Matrix{T}, H::Matrix{T}) where {T}
        p, n, r = nmf_checksize(X, W, H)
        new{T}(Array{T}(undef, p * r), Array{T}(undef, n * r))
    end
end

prepare_state(::BPGUpdCONS{T}, X, W, H) where {T} = BPGUpdCONS_State{T}(X, W, H)
# evaluate_objv(::BPGUpdCONS{T}, s::BPGUpdCONS_State{T}, X, W, H) where T = sqrt(sqL2dist(X, s.WH))

function update_wh!(upd::BPGUpdCONS{T}, s::BPGUpdCONS_State{T}, X, W::Matrix{T}, H::Matrix{T}, jk::Integer) where {T}

    p = size(X, 1)
    n = size(X, 2)
    r = size(W, 2)

    x = W[:]
    y = transpose(H)[:]
    vx = s.vx
    vy = s.vy
    ρ = upd.ρ
    δ = upd.δ
    constraint = upd.constraint

    #compute vx
    vx = 1 / ρ * (norm(x)^2 + norm(y)^2 + 1) * x - (kron(I(r), (W * H - X))) * y

    #compute vy
    vy = 1 / ρ * (norm(x)^2 + norm(y)^2 + 1) * y - transpose(kron(I(r), (W * H - X))) * x

    #project vx and vy
    projectnn!(vx)
    projectnn!(vy)

    #hard_thresholding
    hard_thresholding(vx, constraint)
    hard_thresholding(vy, constraint)

    #compute theta
    m(theta) = (norm(vx)^2 + norm(vy)^2) * theta^3 + theta - ρ
    th = fzero(m, δ, 10^4)

    #compute x^(k+1) and y^(k+1)
    x = th * vx
    y = th * vy

    W = reshape(x, p, r)
    Y = reshape(y, n, r)
    H = copy(transpose(Y))
    return W, H
end