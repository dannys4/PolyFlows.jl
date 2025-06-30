export adjoint_sampling_base_normal
# function LS_stochastic_interpolant(
# 	data::AbstractMatrix{T}, basis::PolyFlowBasis, interp::AbstractStochInterpolant,
# 	time_pts::AbstractVector, time_wts::AbstractVector = ones(length(time_pts));
# 	rng::AbstractRNG = Random.GLOBAL_RNG) where {T}
# 	(; space_basis, time_basis, fmset) = basis
#     N_samples, x_dim = size(data)
# 	x_t_dim = size(fmset,1)
# 	@argcheck x_dim == length(space_basis) DimensionMismatch
# 	@argcheck x_dim + 1 == x_t_dim DimensionMismatch
#     @argcheck length(time_pts) == length(time_wts) DimensionMismatch
#     spaces = initialize_interpolant_spaces(T, N_samples, fmset, rng, true)
#     state, velocity, noise, full_eval_space, univariate_eval_space, LS_matrix, LS_vector = spaces
#     for time_idx in eachindex(time_pts)
#         t, t_wt = time_pts[time_idx], time_wts[time_idx]
#         # Create the state and velocity vectors at this time
#         state_and_velocity!(state, velocity, data, noise, interp, t)
#         # Evaluate the basis for this state
#         velocity_basis_eval_step!(full_eval_space, univariate_eval_space, space_basis, time_basis, fmset, state, t)
#         # Add to the current least-squares matrix and vector
#         mul!(LS_matrix, full_eval_space, full_eval_space', t_wt / N_samples, true)
#         mul!(LS_vector, full_eval_space, velocity, t_wt / N_samples, true)
#     end
#     LS_matrix, LS_vector
# end


raw"""
Consider target distribution ``\pi`` satisfying ``\pi \propto \rho\eta``, where ``\eta`` is a standard MVNormal.
The first argument is the "grad log likelihood" or gradient of the log of Radon--Nikodym derivative of ``\pi`` w.r.t ``\eta``.

# Arguments
- `grad_log_like`: Maps input (N_samples, x_dim) to gradient of output of log-likelihood, (N_samples, x_dim).
"""
function adjoint_sampling_base_normal(grad_log_like, batch_size::Int, basis::PolyFlowBasis,
    interp::AbstractStochInterpolant, time_pts::AbstractVector,
    time_wts::AbstractVector = ones(length(time_pts)); T=Float64, rng::AbstractRNG = Random.GLOBAL_RNG)

    # Reweight for each side
    time_wts_lhs = time_wts ./ interp.eta.(time_pts)
    time_wts_rhs = time_wts .* interp.alpha.(time_pts)
    @argcheck !any(isnan.(time_wts_lhs) .| isinf.(time_wts_lhs)) ArgumentError("Problem with noise schedule")

    spaces = initialize_interpolant_spaces(T, batch_size, basis.fmset, rng, true)
    state, velocity, samples0, full_eval_space, univariate_eval_space, LS_matrix, LS_vector = spaces
    samples1 = similar(samples0)
    randn!(rng, samples1)
    grad_out = grad_log_like(samples1)

    for time_idx in eachindex(time_pts)
        t, t_wt_lhs, t_wt_rhs = time_pts[time_idx], time_wts_lhs[time_idx], time_wts_rhs[time_idx]
        state_and_velocity!(state, velocity, samples1, samples0, interp, t)
        velocity_basis_eval_step!(full_eval_space, univariate_eval_space, basis.space_basis, basis.time_basis, basis.fmset, state, t)
        # Add to the current least-squares matrix and vector
        mul!(LS_matrix, full_eval_space, full_eval_space', t_wt_lhs / batch_size, true)
        mul!(LS_vector, full_eval_space, grad_out, t_wt_rhs / batch_size, true)
    end
    LS_matrix, LS_vector
end