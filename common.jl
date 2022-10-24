# common facilities

# tools to check size

function nmf_checksize(A, X::AbstractMatrix, Y::AbstractMatrix)

    m = size(A, 1)
    r = size(A, 2)
    n = size(X, 2)

    if !(size(X,1) == m && size(Y) == (r, n))
        throw(DimensionMismatch("Dimensions of A, X, and Y are inconsistent."))
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
    objtime::Array{T}
    timeerr::Array{T}

    function Result{T}(X::Matrix{T}, Y::Matrix{T}, niters::Int, converged::Bool, objv, objt, ter) where T
        if size(X, 2) != size(Y, 1)
            throw(DimensionMismatch("Inner dimensions of X and Y mismatch."))
        end
        new{T}(X, Y, niters, converged, objv, objt, ter)
    end
end

# common algorithmic skeleton for iterative updating methods

abstract type NMFUpdater{T} end

function nmf_skeleton!(updater::NMFUpdater{T},
                       A, X::Matrix{T}, Y::Matrix{T},
                       maxiter::Int, verbose::Bool) where T
    objv = convert(T, NaN)
    objt = zeros(T,maxiter)
    ter = zeros(T,maxiter)
    colA = sum(A,dims=2)/size(A,2).+eps(T)
    nA = A.*log.(A./repeat(colA,1,size(A,2)).+eps(T))
    nA = sum(nA)
    # init
    state = prepare_state(updater, A, X, Y)
    preX = Matrix{T}(undef, size(X))
    preY = Matrix{T}(undef, size(Y))
    start = time()
    objv = KLobj(A, X, Y)
    if verbose
        start = time()
        @printf("%-5s    %-13s    %-13s    %-13s\n", "Iter", "Elapsed time", "objv", "objv.change")
        @printf("%5d    %13.6e    %13.6e\n", 0, 0.0, objv)
    end

    # main loop
    converged = false
    t = 0
    while !converged && t < maxiter
        t += 1
        copyto!(preX, X)
        copyto!(preY, Y)

        # update Y
        update_wh!(updater, state, A, X, Y)

        # determine convergence
        #dev = max(maxad(preX, X), maxad(preY, Y))
        #if dev < tol
        #    converged = true
        #end

        # display info
        if verbose
            elapsed = time() - start
            preobjv = objv
            objv = KLobj(A, X, Y)
            @printf("%5d    %13.6e    %13.6e    %13.6e\n",
                t, elapsed, objv, objv - preobjv)
        end
        ter[t] = time() - start
        objt[t] = KLobj(A, X, Y)/nA
    end

    if !verbose
        objv = KLobj(A, X, Y)
    end
    return Result{T}(X, Y, t, converged, objv, objt, ter)
end
