module SCNMF
using StatsBase
using Base
using Statistics
using Printf
using LinearAlgebra
using NonNegLeastSquares
using Random
using RandomizedLinAlg
using SparseArrays
using Distributions
using Roots
using SparseArrays


include("common.jl")
include("utils.jl")

include("initialization.jl")

include("BPG.jl")
include("FBPG.jl")

end # module
