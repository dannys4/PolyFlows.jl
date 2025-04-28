export EulerSampleIntegrator, RK4SampleIntegrator
using Base: size

function get_output(integrator::AbstractSampleIntegrator)
    integrator.out_workspace
end

function Base.size(integrator::AbstractSampleIntegrator)
    size(integrator.out_workspace)
end


struct EulerSampleIntegrator{A<:AbstractArray} <: AbstractSampleIntegrator
    out_workspace::A
end

"""
    EulerSampleIntegrator(ScalarType, dims...)
Integrator for Euler scheme that uses state of dimension (dims...). Space complexity: prod(dims...)
"""
function EulerSampleIntegrator(::Type{T}, dims::Vararg{Int,N}) where {T,N}
    EulerSampleIntegrator{Array{T,N}}(zeros(T, dims...))
end

struct RK4SampleIntegrator{A<:AbstractMatrix} <: AbstractSampleIntegrator
    in_workspace::A
    k_workspace::A
    out_workspace::A
end

"""
    RK4SampleIntegrator(ScalarType, dims...)
Integrator for Runge-Kutta 4 scheme that uses state of dimension (dims...). Space complexity: 3prod(dims...)
"""
function RK4SampleIntegrator(::Type{T}, dims::Vararg{Int,N}) where {T,N}
    in_space = zeros(T, dims...)
    k_space, out_space = zero(in_space), zero(in_space), zero(in_space)
    RK4SampleIntegrator{Array{T,N}}(in_space, k_space, out_space)
end

function integrate!(integrator::EulerSampleIntegrator, du_fcn!, input, t0, dt)
    out_space = get_output(integrator)
    du_fcn!(out_space, input, t0)
    for idx in eachindex(input)
        out_space[idx] = muladd(out_space[idx], dt, input[idx])
    end
    nothing
end

function integrate!(integrator::RK4SampleIntegrator, du_fcn!, input, t0, dt)
    in_space, k_space, out_space = integrator.in_workspace, integrator.k_workspace, integrator.out_workspace
    @argcheck size(in_space) == size(k_space) DimensionMismatch
    @argcheck size(in_space) == size(out_space) Dimen
    for idx in eachindex(input)
        in_space[idx]  = input[idx]
        out_space[idx] = input[idx]
        k_space[idx] = 0.
    end
    # Calculate k1
    du_fcn!(k_space, in_space, t0)
    for idx in eachindex(input)
        in_space[idx]  = muladd(dt/2, k_space[idx], input[idx])
        out_space[idx] = muladd(dt/6, k_space[idx], out_space[idx])
        k_space[idx] = 0.
    end
    # Calculate k2
    du_fcn!(k_space, in_space, t0+dt/2)
    for idx in eachindex(input)
        in_space[idx]  = muladd(dt/2, k_space[idx], input[idx])
        out_space[idx] = muladd(dt/3, k_space[idx], out_space[idx])
        k_space[idx] = 0.
    end
    # Calculate k3
    du_fcn!(k_space, in_space, t0+dt/2)
    for idx in eachindex(input)
        in_space[idx]  = muladd(dt  , k_space[idx], input[idx])
        out_space[idx] = muladd(dt/3, k_space[idx], out_space[idx])
    end
    # Calculate k4
    du_fcn!(k_space, in_space, t0+dt)
    for idx in eachindex(input)
        out_space[idx] = muladd(dt/6, k_space[idx], out_space[idx])
    end
    nothing
end