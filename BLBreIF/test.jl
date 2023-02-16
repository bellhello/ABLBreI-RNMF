include("BLBreIF.jl")
import .BLBreIF
A = rand(Float64, 200, 200)
A = BLBreIF.normalize!(A)
X, Y = BLBreIF.randinit(A, 10)

X₀ = copy(X);
Y₀ = copy(Y);
r₀ = BLBreIF.solve!(BLBreIF.ALBreI{Float64}(runtime=50,
                verbose=false,
                ρ=0.8,
                μ₁=0.005,
                μ₂=0.005), A, X₀, Y₀)

X₁ = copy(X);
Y₁ = copy(Y);
r₁ = BLBreIF.solve!(BLBreIF.ABLBreI{Float64}(runtime=50,
                verbose=false,
                ρ=0.8,
                μ₁=0.005,
                μ₂=0.005), A, X₁, Y₁)

stop₀ = r₀.niters
pic₀ = r₀.objvalue
rt₀ = pic₀[1:stop₀, 1]
obj₀ = pic₀[1:stop₀, 2]

stop₁ = r₁.niters
pic₁ = r₁.objvalue
rt₁ = pic₁[1:stop₁, 1]
obj₁ = pic₁[1:stop₁, 2]

using Plots
using GR
p₁ = Plots.plot(rt₀, obj₀, label="ALBreI")
p₂ = Plots.plot!(rt₁, obj₁, label="ABLBreI")
Plots.plot(p₂; xlabel="time", ylabel="objv",
        xlims=(0.0, 50), ylims=(0.0, 0.2))