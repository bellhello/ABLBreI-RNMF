include("NMF.jl")
A = rand(Float64, 200, 200)
A = BLNMF.normalize!(A)
X, Y = BLNMF.randinit(A, 10)
r = BLNMF.solve!(BLNMF.BLBreIF{Float64}(maxiter = 1000,
    verbose = true,
    tol = 432,
    rho = 0.5,
    mu = 0.001), A, X, Y)