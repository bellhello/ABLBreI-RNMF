include("NMF.jl")
A = rand(Float64, 200, 200)
A = BLNMF.normalize!(A)
X, Y = BLNMF.randinit(A, 10)
r = BLNMF.solve!(BLNMF.BLBreIF{Float64}(maxiter = 100,
    verbose = true,
    rho = 0.2,
    mu = 0.01), A, X, Y)