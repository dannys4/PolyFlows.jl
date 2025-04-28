module PolyFlows
using MultiIndexing, MultivariateExpansions, UnivariateApprox, LinearAlgebra, Random, ArgCheck

abstract type AbstractSampleIntegrator end

include("integrators.jl")
include("interpolants.jl")
include("least_squares.jl")
include("samplers.jl")

end