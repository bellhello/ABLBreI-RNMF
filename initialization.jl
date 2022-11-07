function normalize!(A::AbstractMatrix{T}) where T
    A = A./norm(A, 2)
end

function randinit(A, nrows::Integer, ncols::Integer, r::Integer, T::DataType)
    X = rand(T, nrows, r)
    Y = rand(T, r, ncols)
    XY = X*Y
    X = X./norm(A - XY, 2)
    Y = Y./norm(A - XY, 2)
    return X, Y
end

function randinit(A, r::Integer) 
    m, n = size(A)
    randinit(A, m, n, r, eltype(A))
end
