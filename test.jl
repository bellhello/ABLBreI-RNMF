include("BLNMF.jl")
import .BLNMF
A = rand(Float64, 200, 200)
A = BLNMF.normalize!(A)
X, Y = BLNMF.randinit(A, 10)

X₀ = copy(X);
Y₀ = copy(Y);
r₀ = BLNMF.solve!(BLNMF.LBreIF{Float64}(runtime=60,
                verbose=false,
                ρ=0.2,
                μ=0.01), A, X₀, Y₀)

X₁ = copy(X);
Y₁ = copy(Y);
r₁ = BLNMF.solve!(BLNMF.BLBreIF{Float64}(runtime=60,
                verbose=false,
                ρ=0.2,
                μ=0.01), A, X₁, Y₁)

stop₀ = r₀.niters
pic₀ = r₀.objvalue
rt₀ = pic₀[1:stop₀, 1]
obj₀ = pic₀[1:stop₀, 2]

stop₁ = r₁.niters
pic₁ = r₁.objvalue
rt₁ = pic₁[1:stop1, 1]
obj₁ = pic₁[1:stop1, 2]

using Plots
using GR
p₁ = Plots.plot(rt₀, obj₀, label="LBreIF")
p₂= Plots.plot!(rt₁, obj₁, label="BLBreIF")
Plots.plot(p₂; xlabel="time", ylabel="objv",
        xlims=(0.0, 60.0), ylims=(0.0, 1))