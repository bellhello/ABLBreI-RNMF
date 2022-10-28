include("NMF.jl")
A = rand(Float64, 200, 200)
X, Y = BLNMF.randinit(A, 10)
r = BLNMF.solve!(BLNMF.BLBreIF{Float64}(maxiter = 100,
    verbose = false,
    rho = 0.5,
    mu = 1.0), A, X, Y)