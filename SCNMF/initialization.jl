function normalize!(A::AbstractMatrix{T}) where {T} # Normalize each column of A
    A = A ./ norm(A)
end

function randinit(A, nrows::Integer, ncols::Integer, r::Integer, T::DataType, z::Float64) #
    X = rand(T, nrows, r)
    Y = rand(T, r, ncols)
    
    # Zero out some entries of X and Y
    X = X .* (rand(nrows, r) .> z)
    Y = Y .* (rand(r, ncols) .> z)
    
    # Normalize each column of X and each row of Y
    for i in 1 : r
        X[:, i] = X[:, i]/norm(X[:, i])
        Y[i, :] = Y[i, :]/norm(Y[i, :])
    end

    XY = X * Y
    AXY = A .* XY

    # Normalize X and Y
    X = X .* sqrt(norm(AXY)) ./ (sqrt(2) * norm(XY))
    Y = Y .* sqrt(norm(AXY)) ./ (sqrt(2) * norm(XY))

    return X, Y
end


function randinit(A, r::Integer, z::Float64)
    m, n = size(A)
    randinit(A, m, n, r, eltype(A), z)
end