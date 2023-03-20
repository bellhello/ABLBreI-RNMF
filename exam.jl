include("ABLBreIF/BLBreIF.jl")
import .BLBreIF
include("SCNMF/SCNMF.jl")
import .SCNMF
using Images

ul = "ORL_Faces/s1/1.pgm"
ig = load(ul)

a, b = size(ig)
R = 5

B = Matrix{Float64}(undef, a*b, 400)
for i = 1 : 40
    for j = 1 : 10
        local url = "ORL_Faces/s"*"$i/"*"$j"*".pgm"
        local img = load(url)
        local q = float64.(channelview(img))
        q = q[:]
        global B[:, (i-1)*10 + j] = q
    end
end

A = SCNMF.normalize!(B)

ρ₀ = 0.8
rtime = 60

X, Y = SCNMF.randinit(A, R^2, true)

X₀ = copy(X);
Y₀ = copy(Y);
r₀ = SCNMF.solve!(SCNMF.BPG{Float64}(obj=:cons,
        constraint=0.95,
        runtime=rtime,
        ρ=ρ₀), A, X₀, Y₀)

X₁ = copy(X);
Y₁ = copy(Y);
r₁ = SCNMF.solve!(SCNMF.BBPG{Float64}(obj=:cons,
        constraint=0.95,
        runtime=rtime,
        ρ=ρ₀), A, X₁, Y₁)


X₂ = copy(X);
Y₂ = copy(Y);
r₂ = BLBreIF.solve!(BLBreIF.ALBreI{Float64}(runtime=rtime,
        verbose=false,
        ρ=ρ₀,
        μ₁=0.001,
        μ₂=0.001), A, X₂, Y₂)

X₃ = copy(X);
Y₃ = copy(Y);
r₃ = BLBreIF.solve!(BLBreIF.ABLBreI{Float64}(runtime=rtime,
        verbose=false,
        ρ=ρ₀,
        μ₁=0.001,
        μ₂=0.001), A, X₃, Y₃)

stop₀ = r₀.niters
pic₀ = r₀.objvalue
rt₀ = pic₀[1:stop₀, 1]
obj₀ = pic₀[1:stop₀, 2]

stop₁ = r₁.niters
pic₁ = r₁.objvalue
rt₁ = pic₁[1:stop₁, 1]
obj₁ = pic₁[1:stop₁, 2]

stop₂ = r₂.niters
pic₂ = r₂.objvalue
rt₂ = pic₂[1:stop₂, 1]
obj₂ = pic₂[1:stop₂, 2]

stop₃ = r₃.niters
pic₃ = r₃.objvalue
rt₃ = pic₃[1:stop₃, 1]
obj₃ = pic₃[1:stop₃, 2]

using Plots
using LaTeXStrings
gr()
p₀ = Plots.plot(rt₀, obj₀, label="BPG")
p₁ = Plots.plot!(rt₁, obj₁, label="FBPG")
p₂ = Plots.plot!(rt₂, obj₂, label="ALBreI")
p₃ = Plots.plot!(rt₃, obj₃, label="ABLBreI")
Plots.plot(p₃; xlabel="time", ylabel=L"\frac{1}{2}\Vert A-XY\Vert^2_2",
    xlims=(0, rtime), ylims=(0.0, 1))