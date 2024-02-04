function normalize!(A::AbstractMatrix{T}) where {T} # Normalize each column of A
    A = A ./ norm(A)
end

function find_t(A, X, Y)
    # 定义二分法的边界
    lower_bound = 0.0
    upper_bound = 1.0

    # 定义二分法的精度
    epsilon = 1e-6

    while upper_bound - lower_bound > epsilon
        # 计算中间值
        t = (lower_bound + upper_bound) / 2

        # 计算 A - t * X * Y 的范数
        norm_val = sqL2dist(A, t .* (X * Y))

        if norm_val > 1
            # 如果范数大于1，则将上界调整为中间值
            upper_bound = t
        else
            # 如果范数小于等于1，则将下界调整为中间值
            lower_bound = t
        end
    end

    return (lower_bound + upper_bound) / 2
end

function randinit(A, nrows::Integer, ncols::Integer, r::Integer, T::DataType, z1::Float64, z2::Float64) #
    X = rand(T, nrows, r)
    Y = rand(T, r, ncols)
    
    # Zero out some entries of X and Y
    X = X .* (rand(nrows, r) .< z1)
    Y = Y .* (rand(r, ncols) .< z2)
    
    # Normalize each column of X and each row of Y
    for i in 1 : r
        X[:, i] = X[:, i]/norm(X[:, i])
        Y[i, :] = Y[i, :]/norm(Y[i, :])
    end
    
    t = find_t(A, X, Y)

    # Normalize X and Y
    X = X .* sqrt(t)
    Y = Y .* sqrt(t)

    return X, Y
end


function randinit(A, r::Integer, z1::Float64, z2::Float64)
    m, n = size(A)
    randinit(A, m, n, r, eltype(A), z1, z2)
end