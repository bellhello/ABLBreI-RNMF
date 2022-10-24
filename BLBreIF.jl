mutable struct BLBreIF{T}
    maxiter::Int
    verbose::Bool
    update_Y::Bool
    rho::T                  # L1 regularization coefficient for W
    mu::T                   # L1 regularization coefficient

    function BLBreIF{T}(;maxiter::Integer = 200,
                        verbose::Bool = false,
                        update_Y::Bool = true,
                        rho::Real = 0.5,
                        mu::Real = 1) where T
        maxiter > 1 || throw(ArgumentError("maxiter must be greater than 1."))
        rho < 1 || throw(ArgumentError("rho must be smaller than 1."))
        rho >0 || throw(ArgumentError("maxiter must be greater than 0."))
        new{T}(maxiter, verbose, update_Y, rho)
    end
end

function solve!(alg::BLBreIF{T}, A, X, Y) where {T}
    nmf_skeleton!(BLBreIFUpd{T}(alg.update_Y, sqrt(eps(T))), A, X, Y, alg.maxiter, alg.verbose)
end

struct BLBreIFUpd{T} <: NMFUpdater{T}
    update_Y::Bool
    delta::T
    rho::T
end

struct BLBreIFUpd_State{T}
    XY::Matrix{T}

    function BLBreIFUpd_State{T}(A, X::Matrix{T}, Y::Matrix{T}) where T
        m, n, r = nmf_checksize(A, X, Y)
        new{T}(X * Y)
    end
end

prepare_state(::BLBreIFUpd{T}, A, X, Y) where {T} = BLBreIFUpd_State{T}(A, X, Y)
evaluate_objv(::BLBreIFUpd{T}, s::BLBreIFUpd_State{T}, A, X, Y) where T = convert(T, 0.5) * sqL2dist(A, s.XY)

function update_xy!(upd::BLBreIFUpd{T}, s::BLBreIFUpd_State{T}, A, X:Matrix{T}, Y::Matrix{T}) where T
    rho = upd.