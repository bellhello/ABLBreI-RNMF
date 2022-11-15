function normalize!(A::AbstractMatrix{T}) where T
    A = A./norm(A, 2)
end

function randinit(A, nrows::Integer, ncols::Integer, r::Integer, T::DataType, z::Bool)
    X = rand(T, nrows, r)
    Y = transpose(X)
    XY = X*Y
    AXY = A.*XY
    X = X.*sqrt(2*sum(AXY))./norm(XY, 2)
    Y = copy(X')
    if !z
        X = zeros(nrows, r)
        Y = zeros(r, ncols)
    end
    return X, Y
end

function randinit(A, r::Integer, z::Bool=true) 
    m, n = size(A)
    randinit(A, m, n, r, eltype(A), z)
end