function shrink(x, y)
    if x > y
        dist = x -y
    elseif x >= -y
        dist = 0
    else
        dist = x + y
    end
    return dist
end

function LinBregman(X, f, n)
    u = zeros(n, 1)
    v = zeros(n, 1)
    delta = .01
    mu = 1000
    epsilon = 10^(-5)
    k = 0
    while norm(f-X*u, 2)/norm(f, 2) > epsilon
        k = k + 1
        v = v + transpose(X)*(f - X*u)
        u = delta*shrink(v, mu)
        r[k] = log(norm(f-X*u, 2))
    end
    return (k, u, r)
end
