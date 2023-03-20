# common facilities

# tools to check size

function nmf_checksize(X, W::AbstractMatrix, H::AbstractMatrix)

    p = size(X, 1)
    n = size(X, 2)
    r = size(W, 2)

    if !(size(W, 1) == p && size(H) == (r, n))
        throw(DimensionMismatch("Dimensions of X, W, and H are inconsistent."))
    end

    return (p, n, r)
end


# the result type

struct Result{T}
    W::Matrix{T}
    H::Matrix{T}
    niters::Int
    objvalue::Matrix{T}

    function Result{T}(W::Matrix{T}, H::Matrix{T}, niters::Int, objvalue::Matrix{T}) where {T}
        if size(W, 2) ≠ size(H, 1)
            throw(DimensionMismatch("Inner dimensions of W and H mismatch."))
        end
        new{T}(W, H, niters, objvalue)
    end
end

# common algorithmic skeleton for iterative updating methods

abstract type NMFUpdater{T} end

function nmf_skeleton!(updater::NMFUpdater{T},
    X, W::Matrix{T}, H::Matrix{T},
    runtime::Int) where {T}
    # init
    state = prepare_state(updater, X, W, H)
    r = size(W, 2)
    t = 1
    objv = zeros(Float64, 10^6, 2)
    objv[t, 2] = sqL2dist(X, W * H)
    start = time()
    while objv[t, 1] ≤ runtime
        # update
        j = mod(t - 1, r) + 1
        v = update_wh!(updater, state, X, W, H, j)
        #record the time
        objv[t+1, 1] = time() - start
        W = v[1]
        H = v[2]
        objv[t+1, 2] = sqL2dist(X, W * H)
        t += 1
    end

    return Result{T}(W, H, t, objv)
end
