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
    objvalue::Matrix{T}

    function Result{T}(X::Matrix{T}, Y::Matrix{T}, niters::Int, converged::Bool, objvalue) where {T}
        if size(X, 2) ≠ size(Y, 1)
            throw(DimensionMismatch("Inner dimensions of X and Y mismatch."))
        end
        new{T}(X, Y, niters, converged, objvalue)
    end
end

# common algorithmic skeleton for iterative updating methods

abstract type NMFUpdater{T} end

function nmf_skeleton!(updater::NMFUpdater{T},
    A, X::Matrix{T}, Y::Matrix{T},
    runtime::Int, verbose::Bool, τ) where {T}
    objv = convert(T, NaN)
    objvalue = zeros(Float64, 10^6, 2)
    m, n, r = nmf_checksize(A, X, Y)
    # init
    state = prepare_state(updater, A, X, Y)
    Px = ones(T, m, r)
    Py = ones(T, r, n)
    preX = Matrix{T}(undef, size(X))
    preY = Matrix{T}(undef, size(Y))

    if verbose
        objv = convert(T, 0.5) * sqL2dist(A, X * Y)
        @printf("%-5s    %-13s    %-13s    %-13s    %-13s\n", "Iter", "Elapsed time", "objv", "objv.change", "(X & Y).change")
        @printf("%5d    %13.6e    %13.6e\n", 0, 0.0, objv)
    end

    # main loop
    converged = false
    k = 1
    objvalue[k, 2] = sqL2dist(A, X * Y)
    start = time()
    while objvalue[k, 1] ≤ runtime
        copyto!(preX, X)
        copyto!(preY, Y)

        # update 
        jₖ = mod(k - 1, r) + 1
        v = update_xy!(updater, state, A, X, Y, Px, Py, jₖ)
        elapsed = time() - start
        objvalue[k+1, 1] = elapsed
        X = v[1]
        Y = v[2]
        Px = v[3]
        Py = v[4]
        objvalue[k+1, 2] = sqL2dist(A, X * Y)
        # determine convergence
        dev = max(maxad(preX, X), maxad(preY, Y))
        if dev < τ
            converged = true
        end

        # display info
        if verbose
            preobjv = objv
            objv = sqL2dist(A, X * Y)
            @printf("%5d    %13.6e    %13.6e    %13.6e    %13.6e\n",
                k, elapsed, objv, objv - preobjv, dev)
        end
        k += 1
    end

    if !verbose
        objv = sqL2dist(A, X * Y)
    end
    return Result{T}(X, Y, k, converged, objvalue)
end
