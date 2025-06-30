module PolyFlows
using MultiIndexing, MultivariateExpansions, UnivariateApprox, LinearAlgebra, Random, ArgCheck
using UnivariateApprox: UnivariateBasis

abstract type AbstractSampleIntegrator end

include("interpolants.jl")
include("basis.jl")
include("integrators.jl")
include("least_squares.jl")
include("samplers.jl")
include("adjoint_sampling.jl")

end