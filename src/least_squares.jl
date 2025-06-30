export create_LS_stoch_interp

"""
    create_full_LS_stoch_interp(data::AbstractMatrix, space_basis::MultivariateBasis, time_basis::UnivariateBasis,
    fmset::FixedMultiIndexSet, interp::AbstractStochInterpolant, time_pts, time_wts; [rng])
Create a least-squares system for the stochastic interpolant assuming the same set of basis functions for every output

If `N=length(fmset)`, then this returns a `NxN` matrix and a `Nxd` vector, where
`d` is the dimension of the state (keeping in mind that fmset is d+1 dimensional due to time)

# Arguments
- `data` (M,d) samples drawn from target
- `space_basis` d-dimensional basis for the space variables
- `time_basis` univariate basis for the time variable
- `fmset` (N,d+1) set of multi-indices
- `interp` stochastic interpolant
- `time_pts` (T) set of points to integrate loss over in time
- `time_wts` (T) set of weights for time integration rule. Defaults to uniform.
- `[rng]` (optional) RNG from Random
"""
function create_full_LS_stoch_interp(data::AbstractMatrix{T}, space_basis::MultivariateBasis{x_dim},
    time_basis::UnivariateBasis, fmset::FixedMultiIndexSet{x_t_dim}, interp::AbstractStochInterpolant,
    time_pts::AbstractVector, time_wts::AbstractVector=ones(length(time_pts));
    rng::AbstractRNG=Random.GLOBAL_RNG) where {T, x_dim, x_t_dim}

	@argcheck x_dim + 1 == x_t_dim DimensionMismatch
    @argcheck length(time_pts) == length(time_wts) DimensionMismatch
    N_samples = size(data, 1)
    spaces = initialize_interpolant_spaces(T, N_samples, fmset, rng, true)
    state, velocity, noise, full_eval_space, univariate_eval_space, LS_matrix, LS_vector = spaces
    for time_idx in eachindex(time_pts)
        t, t_wt = time_pts[time_idx], time_wts[time_idx]
        # Create the state and velocity vectors at this time
        state_and_velocity!(state, velocity, data, noise, interp, t)
        # Evaluate the basis for this state
        velocity_basis_eval_step!(full_eval_space, univariate_eval_space, space_basis, time_basis, fmset, state, t)
        # Add to the current least-squares matrix and vector
        mul!(LS_matrix, full_eval_space, full_eval_space', t_wt / N_samples, true)
        mul!(LS_vector, full_eval_space, velocity, t_wt / N_samples, true)
    end
    LS_matrix, LS_vector
end