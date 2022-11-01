# common facilities

# tools to check size

function nmf_checksize(A, X::AbstractMatrix, Y::AbstractMatrix)

    m = size(A, 1)
    r = size(X, 2)
    n = size(A, 2)

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

    function Result{T}(X::Matrix{T}, Y::Matrix{T}, niters::Int, converged::Bool, objv) where T
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
                       maxiter::Int, verbose::Bool) where T
    objv = convert(T, NaN)
    
    # init
    state = prepare_state(updater, A, X, Y)
    preX = Matrix{T}(undef, size(X))
    preY = Matrix{T}(undef, size(Y))
    if verbose
        start = time()
        @printf("%-5s    %-13s    %-13s    %-13s\n", "Iter", "Elapsed time", "objv", "objv.change")
        @printf("%5d    %13.6e    %13.6e\n", 0, 0.0, objv)
    end

    # main loop
    converged = false
    k = 0
    while !converged && k < maxiter
        k += 1
        copyto!(preX, X)
        copyto!(preY, Y)

        # update 
        j_k = mod(k, size(X,2)) + 1
        update_xy!(updater, state, A, X, Y, j_k)

        # determine convergence
        #dev = max(maxad(preX, X), maxad(preY, Y))
        #if dev < tol
        #    converged = true
        #end

        # display info
        if verbose
            elapsed = time() - start
            preobjv = objv
            objv = evaluate_objv(updater, state, A, X, Y)
            @printf("%5d    %13.6e    %13.6e    %13.6e\n",
                t, elapsed, objv, objv - preobjv)
        end
    end

    if !verbose
        objv = evaluate_objv(updater, state, A, X, Y)
    end
    return Result{T}(X, Y, k, converged, objv)
end
