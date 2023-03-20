using Images


ul = "ORL_Faces/s1/1.pgm"
ig = load(ul)

a = 60
b = 60
R = 5

B = Matrix{Float64}(undef, a * b, 400)
for i = 1:40
    for j = 1:10
        local url = "ORL_Faces/s" * "$i/" * "$j" * ".pgm"
        local img = load(url)
        local q = float64.(img)
        q = q[46:105,17:76][:]
        global B[:, (i-1)*10+j] = q
    end
end

# 计算矩阵B中零元素的个数
s = 0
for i = 1 : size(B, 1)
    for j = 1 : size(B, 2)
        if B[i, j] == 0
            s += 1
        end
    end
end


include("ABLBreIF/BLBreIF.jl")

A = BLBreIF.normalize!(B)
X, Y = BLBreIF.randinit(A, R^2, 0.8)
r = BLBreIF.solve!(BLBreIF.ABLBreI{Float64}(runtime=600,
        verbose=true,
        ρ=0.6,
        μ₁=0.2,
        μ₂=0.2), A, X, Y)

X = norm(B) .* X
#import NMF
#W, H = NMF.randinit(X, R^2)
#r = NMF.solve!(NMF.MultUpdate{Float64}(obj=:mse, maxiter=100), X, W, H)

ViewW = Array{Gray{Float64},2}(undef, a * R, b * R)
ViewH = Array{Gray{Float64},2}(undef, R, R)

#W = r.W*sqrt(sqrt(s))

for k = 0 : R^2-1
    local w = X[:, k+1]
    # local h = Y[k+1, 2]
    local m = reshape(w, a, b)
    local m = float(Gray.(m))
    # local n = float(Gray.(1 .- h))
    local i = div(k, R)
    local j = mod(k, R)
    ViewW[i*a+1:(i+1)*a, j*b+1:(j+1)*b] = m
    # ViewH[i+1, j+1] = n
end

ViewW
