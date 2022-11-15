include("NMF.jl")
A = rand(Float64, 200, 200)
A = BLNMF.normalize!(A)
X, Y = BLNMF.randinit(A, 10)

X0 = copy(X);
Y0 = copy(Y);
r0 = BLNMF.solve!(BLNMF.LBreIF{Float64}(runtime=300,
        verbose=false,
        rho=0.2,
        mu=0.01), A, X, Y)

X1 = copy(X);
Y1 = copy(Y);
r1 = BLNMF.solve!(BLNMF.BLBreIF{Float64}(runtime=300,
        verbose=false,
        rho=0.2,
        mu=0.01), A, X, Y)

stop0 = r0.niters
pic0 = r0.objvalue
rt0 = pic0[1:stop0, 1]
obj0 = pic0[1:stop0, 2]

stop1 = r1.niters
pic1 = r1.objvalue
rt1 = pic1[1:stop1, 1]
obj1 = pic1[1:stop1, 2]

using Plots
using GR
p1 = Plots.plot(rt0, obj0, label="LBreIF")
p2 = Plots.plot!(rt1, obj1, label="BLBreIF")
Plots.plot(p2; xlabel="time", ylabel="objv",
    xlims=(0.0, 300.0), ylims=(0.0, 1))