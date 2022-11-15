include("NMF.jl")
A = rand(Float64, 200, 200)
A = BLNMF.normalize!(A)
X, Y = BLNMF.randinit(A, 10)
r = BLNMF.solve!(BLNMF.LBreIF{Float64}(maxiter = 30,
    verbose = true,
    tol = 432,
    rho = 0.2,
    mu = 0.001), A, X, Y)