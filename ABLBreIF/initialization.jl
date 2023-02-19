function normalize!(A::AbstractMatrix{T}) where {T}
    A = A ./ norm(A)
end

function randinit(A, nrows::Integer, ncols::Integer, r::Integer, T::DataType, z::Bool)
    X = rand(T, nrows, r)
    Y = rand(T, r, ncols)

    #Normalize each column of X and each row of Y
    for i in 1:length(X[1, :])
        X[:, i] = X[:, i]/norm(X[:, i])
    end
    for j in 1:length(Y[:, 1])
        Y[j, :] = Y[j, :]/norm(Y[j, :])
    end

    XY = X * Y
    AXY = A .* XY

    #Normalize X and Y
    X = X .* sqrt(sum(AXY)) ./ norm(XY, 2)
    Y = Y .* sqrt(sum(AXY)) ./ norm(XY, 2)

    if !z
        X .= 0
        Y .= 0
    end
    
    return X, Y
end


function randinit(A, r::Integer, z::Bool=true)
    m, n = size(A)
    randinit(A, m, n, r, eltype(A), z)
end