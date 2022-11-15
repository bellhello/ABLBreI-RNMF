include("NMF.jl")
A = rand(Float64, 200, 200)
A = BLNMF.normalize!(A)
X, Y = BLNMF.randinit(A, 10)
X0 = copy(X); Y0 = copy(Y);
r = BLNMF.solve!(BLNMF.BLBreIF{Float64}(maxiter = 1000,
    verbose = true,
    tol = 432,
    rho = 0.2,
    mu = 0.001), A, X0, Y0)