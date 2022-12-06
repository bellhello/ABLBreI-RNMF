include("BLNMF/BLNMF.jl")
import .BLNMF
include("SCNMF/SCNMF.jl")
import .SCNMF

using SparseArrays
A = sprand(Float64, 500, 500, 0.8)
A = SCNMF.normalize!(A)

ρ₀ = 0.2
μ₀ = 0.1
rtime = 30

X, Y = SCNMF.randinit(A, 10, true)

X₀ = copy(X);
Y₀ = copy(Y);
r₀ = SCNMF.solve!(SCNMF.BPG{Float64}(obj=:cons,
                constrain=1 - ρ₀,
                runtime=rtime,
                ρ=ρ₀), A, X₀, Y₀)

X₁ = copy(X);
Y₁ = copy(Y);
r₁ = SCNMF.solve!(SCNMF.FBPG{Float64}(obj=:cons,
                constrain=1 - ρ₀,
                runtime=rtime,
                ρ=ρ₀), A, X₁, Y₁)


X₂ = copy(X);
Y₂ = copy(Y);
r₂ = BLNMF.solve!(BLNMF.LBreIF{Float64}(runtime=rtime,
                verbose=false,
                ρ=ρ₀,
                μ=μ₀), A, X₂, Y₂)

X₃ = copy(X);
Y₃ = copy(Y);
r₃ = BLNMF.solve!(BLNMF.BLBreIF{Float64}(runtime=rtime,
                verbose=false,
                ρ=ρ₀,
                μ=μ₀), A, X₃, Y₃)

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
# p₁ = Plots.plot!(rt₁, obj₁, label="FBPG")
# p₂ = Plots.plot!(rt₂, obj₂, label="LBreIF")
# p₃ = Plots.plot!(rt₃, obj₃, label="BLBreIF")
# Plots.plot(p₃; xlabel="time", ylabel=L"\frac{1}{2}\Vert A-XY\Vert^2_2",
#     xlims=(0, rtime), ylims=(0.0, 1))

function speed()
        lines(rt₀, obj₀; color="#4063D8", label="BPG", linewidth=2, linestyle=:dash,
                figure=(; figure_padding=5, resolution=(1200, 800), font="sans",
                        backgroundcolor=:white, fontsize=32),
                axis=(; xlabel="time", ylabel=L"\frac{1}{2}\Vert A-XY\Vert^2_2", yscale=log10, title="Objective Value",
                        xgridstyle=:dash, ygridstyle=:dash))
        lines!(rt₁, obj₁; color="#389826", linewidth=2, linestyle=:dash,
                label="FBPG")
        lines!(rt₂, obj₂; color="#9558B2", linewidth=2, linestyle=nothing,
                label="LBreI")
        lines!(rt₃, obj₃; color="#CB3C33", linewidth=2, linestyle=nothing,
                label="BLBreI")
        limits!(-0.1, rtime, 0.1, 1.0)
        axislegend("Algorithms"; merge=true)
        current_figure()
end

with_theme(speed, theme = theme_ggplot2())
save("plot/exam.png", speed())
