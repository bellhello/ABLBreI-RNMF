function normalize!(A::AbstractMatrix{T}) where T
    A = A./norm(A, 2)
end

function randinit(A, nrows::Integer, ncols::Integer, r::Integer, T::DataType)
    X = rand(T, nrows, r)
    Y = rand(T, r, ncols)
    for i = 1:r
        X[:,i] = X[:,i]/norm(X[:,i], 2)
        Y[i,:] = Y[i,:]/norm(Y[i,:], 2)
    end
    XY = X*Y
    alpha = sqrt(sum(A)/sum(XY))
    X = alpha*X 
    Y = alpha*Y
    return X, Y
end

function randinit(A, r::Integer) 
    m, n = size(A)
    randinit(A, m, n, r, eltype(A))
end
