include("SCNMF.jl")
import .SCNMF
using SparseArrays
A = sprand(Float64, 200, 200, 0.8)
A = SCNMF.normalize!(A)

X, Y = SCNMF.randinit(A, 10, true)
X₀ = copy(X);
Y₀ = copy(Y);
r₀ = SCNMF.solve!(SCNMF.BPG{Float64}(obj=:cons,
                constraint=0.95,
                runtime=60,
                ρ=0.2), A, X₀, Y₀)

X₁ = copy(X);
Y₁ = copy(Y);
r₁ = SCNMF.solve!(SCNMF.FBPG{Float64}(obj=:cons,
                constraint=0.95,
                runtime=60,
                ρ=0.2), A, X₁, Y₁)

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
p₁ = Plots.plot(rt₀, obj₀, label="BPG")
p₂ = Plots.plot!(rt₁, obj₁, label="FBPG")
Plots.plot(p₂; xlabel="time", ylabel="objv",
        xlims=(0.0, 60.0), ylims=(0.0, 0.5))