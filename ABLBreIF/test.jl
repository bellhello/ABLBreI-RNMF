include("BLBreIF.jl")
import .BLBreIF
A = rand(Float64, 200, 200)
A = BLBreIF.normalize!(A)
X, Y = BLBreIF.randinit(A, 10)

X₀ = copy(X);
Y₀ = copy(Y);
r₀ = BLBreIF.solve!(BLBreIF.ALBreI{Float64}(runtime=30,
                verbose=false,
                ρ=0.8,
                μ₁=0.05,
                μ₂=0.05), A, X₀, Y₀)

X₁ = copy(X);
Y₁ = copy(Y);
r₁ = BLBreIF.solve!(BLBreIF.ABLBreI{Float64}(runtime=30,
                verbose=false,
                ρ=0.8,
                μ₁=0.05,
                μ₂=0.05), A, X₁, Y₁)

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
using LaTeXStrings

p₁ = Plots.plot(rt₀, obj₀, label="ALBreI")
p₂ = Plots.plot!(rt₁, obj₁, label="ABLBreI")
Plots.plot(p₂; xlabel="Time(sec)", ylabel=L"\Vert A-XY\Vert_F",
        xlims=(-0.1, 50), ylims=(0.2, 0.4))