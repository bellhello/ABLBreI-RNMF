using Images


ul = "ORL_Faces/s1/1.pgm"
ig = load(ul)

a, b = size(ig)
R = 5

B = Matrix{Float64}(undef, a * b, 400)
for i = 1:40
    for j = 1:10
        local url = "ORL_Faces/s" * "$i/" * "$j" * ".pgm"
        local img = load(url)
        local q = float64.(channelview(img))
        q = q[:]
        global B[:, (i-1)*10+j] = q
    end
end

include("BLNMF/BLNMF.jl")

A = BLNMF.normalize!(B)
X, Y = BLNMF.randinit(A, R^2)
r = BLNMF.solve!(BLNMF.BLBreIF{Float64}(runtime=30,
        verbose=true,
        ρ=0.8,
        μ=0.01), A, X, Y)

X = norm(B) .* X
#import NMF
#W, H = NMF.randinit(X, R^2)
#r = NMF.solve!(NMF.MultUpdate{Float64}(obj=:mse, maxiter=100), X, W, H)

ViewW = Array{Gray{Float64},2}(undef, a * R, b * R)
ViewH = Array{Gray{Float64},2}(undef, R, R)

#W = r.W*sqrt(sqrt(s))

for k = 0:R^2-1
    local w = X[:, k+1]
    local h = Y[k+1, 2]
    local m = reshape(w, a, b)
    local m = float(Gray.(m))
    local n = float(Gray.(h))
    local i = div(k, R)
    local j = mod(k, R)
    ViewW[i*a+1:(i+1)*a, j*b+1:(j+1)*b] = m
    ViewH[i+1, j+1] = n
end

ViewW
