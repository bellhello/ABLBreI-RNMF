include("BLBreIF/BLBreIF.jl")
include("SCNMF/SCNMF.jl")
using .BLBreIF
using .SCNMF

using SparseArrays
A = sprand(Float64, 500, 500, 0.95)
A = SCNMF.normalize!(A)

ρ₀ = 0.8
μ₀ = 0.01
rtime = 120

X, Y = SCNMF.randinit(A, 10, true)

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
                μ₁=μ₀,
                μ₂=μ₀), A, X₂, Y₂)

X₃ = copy(X);
Y₃ = copy(Y);
r₃ = BLBreIF.solve!(BLBreIF.ABLBreI{Float64}(runtime=rtime,
                verbose=false,
                ρ=ρ₀,
                μ₁=μ₀,
                μ₂=μ₀), A, X₃, Y₃)

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

# using Plots
# using LaTeXStrings
using CairoMakie
using Makie
using LaTeXStrings
using GLMakie
using Colors
using AlgebraOfGraphics
CairoMakie.activate!()

# gr()
# p₀ = Plots.plot(rt₀, obj₀, label="BPG")
# p₁ = Plots.plot!(rt₁, obj₁, label="BBPG")
# p₂ = Plots.plot!(rt₂, obj₂, label="LBreIF")
# p₃ = Plots.plot!(rt₃, obj₃, label="ABLBreI")
# Plots.plot(p₃; xlabel="time", ylabel=L"\frac{1}{2}\Vert A-XY\Vert^2_2",
#     xlims=(0, rtime), ylims=(0.0, 1))

function speed()
        lines(rt₀, sqrt.(2*obj₀); color="#4063D8", label="BPG", linewidth=3, linestyle=:dashdot,
                figure=(; figure_padding=50, resolution=(1200, 800), font="sans",
                        backgroundcolor=:white, fontsize=32),
                axis=(; xlabel="Time(sec)", ylabel=L"\Vert A-XY^T\Vert_F",
                        #yscale=log10,
                        xgridstyle=:dash, ygridstyle=:dash))
        lines!(rt₁, sqrt.(2*obj₁); color="#389826", linewidth=3, linestyle=:dashdot,
                label="BBPG")
        lines!(rt₂, sqrt.(2*obj₂); color="#FFC633", linewidth=2, linestyle=:solid,
                label="ALBreI")
        lines!(rt₃, sqrt.(2*obj₃); color="#CB3C33", linewidth=2, linestyle=:solid,
                label="ABLBreI")
        limits!(0, 120, 0.6, 0.7)
        axislegend("Algorithms"; merge=true)
        current_figure()
end

with_theme(speed, theme=theme_dark())
save("plot/exam.png", speed())
