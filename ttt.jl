include("NMF.jl")
using LinearAlgebra
using Roots
using StatsBase
A = rand(Float64, 200, 200)
A = BLNMF.normalize!(A)
X, Y = BLNMF.randinit(A, 10)
k = 0
function nmf_checksize(A, X::AbstractMatrix, Y::AbstractMatrix)

    m = size(A, 1)
    n = size(A, 2)
    r = size(X, 2)


    if !(size(X, 1) == m && size(Y) == (r, n))
        throw(DimensionMismatch("Dimensions of A, X and Y are inconsistent."))
    end

    return (m, n, r)
end
m, n, r = nmf_checksize(A, X, Y)
j_k = mod(k, r) + 1
rho = 0.2
mu = 0.01
T = Float64

XY = X*Y
function soft_thresholding(A::AbstractArray{T}, s::Real) where T
    B = copy(A)
    n = length(B)
    @inbounds for i = 1 : n
        x = sign(B[i])*max(abs(B[i]) - s, 0)
        B[i] = x
    end
    return B
end
    x = X[:]
    y = transpose(Y)[:]
px = zeros(T, m* r)
py = zeros(T, r* n)
    # update x_j
    Vx = 1/rho*(norm(x)^2 + norm(y)^2 + 1)*x + mu*px - (kron(I(r), (X*Y - A)))*y
    v = soft_thresholding(rho*Vx, rho*mu)
    f(t) = norm(v, 2)^2*t^3 + t - 1
    t_0 = fzero(f, 0)
    x_1 = t_0*v
    px = px - 1/(rho*mu)*((norm(x_1)^2 + norm(y)^2 + 1)*x_1 - (norm(x)^2 + norm(y)^2 + 1)*x + rho*(kron(I(r), (X*Y - A)))*y)

    X1 = reshape(x_1, m, r)
    XY1 = X1*Y
    # update y_j
    Vy = 1/rho*(norm(x_1)^2 + norm(y)^2 + 1)*y+ mu*py - transpose(kron(I(r), (XY1 - A)))*x_1
    v = soft_thresholding(rho*Vy, rho*mu)
    g(t) = norm(v)^2*t^3 + t - 1
    t_0 = fzero(g, 0)
    y_1 = t_0*v
    py = py - 1/(rho*mu)*((norm(x_1)^2 + norm(y_1)^2 + 1)*y_1 - (norm(x_1)^2 + norm(y)^2 + 1)*y + rho*transpose(kron(I(r), (XY1 - A)))*x_1)

    X = reshape(x_1, m, r)
    Y = copy(transpose(reshape(y_1, n, r)))
    convert(T,0.5)*sqL2dist(A, X*Y)