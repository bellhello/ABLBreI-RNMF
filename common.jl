# common facilities

# tools to check size

function nmf_checksize(A, X::AbstractMatrix, Y::AbstractMatrix)

    m = size(A, 1)
    n = size(A, 2)
    r = size(X, 2)


    if !(size(X, 1) == m && size(Y) == (r, n))
        throw(DimensionMismatch("Dimensions of A, X and Y are inconsistent."))
    end

    return (m, n, r)
end

# the result type

struct Result{T}
    X::Matrix{T}
    Y::Matrix{T}
    niters::Int
    converged::Bool
    objvalue::T

    function Result{T}(X::Matrix{T}, Y::Matrix{T}, niters::Int, converged::Bool, objv) where {T}
        if size(X, 2) != size(Y, 1)
            throw(DimensionMismatch("Inner dimensions of X and Y mismatch."))
        end
        new{T}(X, Y, niters, converged, objv)
    end
end

# common algorithmic skeleton for iterative updating methods

abstract type NMFUpdater{T} end

function nmf_skeleton!(updater::NMFUpdater{T},
    A, X::Matrix{T}, Y::Matrix{T},
    maxiter::Int, verbose::Bool, tol) where {T}
    objv = convert(T, NaN)

    m, n, r = nmf_checksize(A, X, Y)
    # init
    state = prepare_state(updater, A, X, Y)
    Px = zeros(T, m, r)
    Py = zeros(T, r, n)
    preX = Matrix{T}(undef, size(X))
    preY = Matrix{T}(undef, size(Y))
    if verbose
        start = time()
        objv = convert(T,0.5)*sqL2dist(A, X*Y)
        @printf("%-5s    %-13s    %-13s    %-13s    %-13s\n", "Iter", "Elapsed time", "objv", "objv.change", "(X & Y).change")
        @printf("%5d    %13.6e    %13.6e\n", 0, 0.0, objv)
    end

    # main loop
    converged = false
    k = 0
    while !converged && k <= maxiter
        copyto!(preX, X)
        copyto!(preY, Y)

        # update 
        j_k = mod(k, r) + 1
        v = update_xy!(updater, state, A, X, Y, Px, Py, j_k)
        X = v[1]
        Y = v[2]
        Px = v[3]
        Py = v[4]
        # determine convergence
        dev = max(maxad(preX, X), maxad(preY, Y))
        if dev < tol
            converged = true
        end

        # display info
        if verbose
            elapsed = time() - start
            preobjv = objv
            objv = convert(T,0.5)*sqL2dist(A, X*Y)
            @printf("%5d    %13.6e    %13.6e    %13.6e    %13.6e\n",
                k, elapsed, objv, objv - preobjv, dev)
        end
        k += 1
    end

    if !verbose
        objv = convert(T,0.5)*sqL2dist(A, X*Y)
    end
    return Result{T}(X, Y, k, converged, objv)
end
