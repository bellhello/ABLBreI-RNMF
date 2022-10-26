function randinit(X, nrows::Integer, ncols::Integer, r::Integer, T::DataType)
    W = rand(T, nrows, r)
    H = rand(T, r, ncols)
    for i = 1:r
        W[:,i] = W[:,i]/norm(W[:,i], 2)
        H[i,:] = H[i,:]/norm(H[i,:], 2)
    end
    WH = W*H
    alpha = sqrt(sum(X)/sum(WH))
    W = alpha*W 
    H = alpha*H
    return W, H
end

function randinit(X, r::Integer) 
    p, n = size(X)
    randinit(X, p, n, r, eltype(X))
end
