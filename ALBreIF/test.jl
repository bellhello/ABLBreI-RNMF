include("ALBreIF.jl")
import .ALBreIF
using SparseArrays
A = sprand(Float64, 200, 10, 0.2) * sprand(Float64, 10, 200, 1.0)
A = ALBreIF.normalize!(A)
X, Y = ALBreIF.randinit(A, 10, 0.2, 1.0) # 10 is the rank

X₀ = copy(X);
Y₀ = copy(Y);
r₀ = ALBreIF.solve!(ALBreIF.ALBreI{Float64}(runtime=60,
                verbose=true,
                ρ=0.08,
                μ₁=0.001,
                μ₂=0.0), A, X₀, Y₀)

X₁ = copy(X);
Y₁ = copy(Y);
r₁ = ALBreIF.solve!(ALBreIF.ABLBreI{Float64}(runtime=60,
                verbose=true,
                ρ=0.08,
                μ₁=0.001,
                μ₂=0.0), A, X₁, Y₁)

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
Plots.plot(p₂; xlabel="Time(sec)", ylabel=L"\Vert A-XY\Vert_F", yscale=:log10, xlims=(-0.1, 60), ylims=(10^(-2), 1.0))