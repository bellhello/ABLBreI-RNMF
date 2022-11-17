function normalize!(A::AbstractMatrix{T}) where {T}
    A = A ./ norm(A)
end

function randinit(A, nrows::Integer, ncols::Integer, r::Integer, T::DataType, z::Bool)
    X = rand(T, nrows, r)
    Y = rand(T, r, ncols)
    X = X./norm(X)
    Y = Y./norm(Y)
    XY = X * Y
    AXY = A .* XY
    X = X .* sqrt(2 * sum(AXY)) ./ norm(XY, 2)
    Y = Y .* sqrt(2 * sum(AXY)) ./ norm(XY, 2)
    if !z
        X = zeros(T, nrows, r)
        Y = zeros(T, r, ncols)
    end
    return X, Y
end

function randinit(A, r::Integer, z::Bool=true)
    m, n = size(A)
    randinit(A, m, n, r, eltype(A), z)
end