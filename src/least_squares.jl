export create_LS_stoch_interp

"""
	velocity_basis_eval_step!(full_eval_space, univariate_eval_space, space_basis, time_basis, fmset, state, time)
Evaluate \$\\{\\Phi(X_t^{(j)},t)\\}_{j=1}^M\\subset \\mathbb{R}^{N}\$ for fixed \$t\$.

# Arguments
- `full_eval_space`: Matrix (N,M)
- `univariate_eval_space::NTuple{d+1}`: Each matrix is `(p_j,M)`, where `p_j` is the maximum degree of basis function in dim `j`
- `space_basis::MultivariateBasis{d}`
- `time_basis::UnivariateBasis`
- `fmset::FixedMultiIndex{d+1}`
- `state` Matrix (M,d)
- `time` Scalar
"""
function velocity_basis_eval_step!(full_eval_space::EM, univariate_eval_space::NTuple{x_t_dim,UM},
    space_basis::MultivariateBasis{x_dim}, time_basis::UnivariateBasis, fmset::FixedMultiIndexSet{x_t_dim},
    state::SM, time::Number) where {x_dim, x_t_dim, T<:Number,EM<:AbstractMatrix{T},UM<:AbstractMatrix{T},SM<:AbstractMatrix{T}}

    N_samples = size(state, 1)
    num_coeff = length(fmset)
    # Dimension checking
    @argcheck x_dim + 1 == x_t_dim DimensionMismatch
    @argcheck size(state, 2) == x_dim DimensionMismatch
    @argcheck size(full_eval_space) == (num_coeff, N_samples) DimensionMismatch
    for (idx,space) in enumerate(univariate_eval_space)
        @argcheck size(space) == (fmset.max_orders[idx] + 1, N_samples) DimensionMismatch
    end

    # Evaluate the univariate basis at the data points
    Evaluate!(univariate_eval_space[1:end-1], space_basis, state)
    time_eval_space = univariate_eval_space[end]
    Evaluate!(@view(time_eval_space[:,1:1]), time_basis, [time])
    @inbounds for col_idx in 2:N_samples
        time_eval_space[:,col_idx] .= time_eval_space[:,1]
    end

    # Evaluate the full basis at the data points
    Evaluate!(full_eval_space, fmset, univariate_eval_space)
    nothing
end

"""
	state_and_velocity!(state, velocity, data, noise, interp, time)
Get \$X_t = \\alpha(t) X_1 + \\beta(t) X_0\$ as well as \$\\dot{X}_t\$, where \$X_1\$ is drawn from the target and \$X_0\$ from the reference.

Output results into `state` and `velocity`.

# Arguments
- `state`: Matrix (M, d) for \$X_t\$
- `velocity`: Matrix (M, d) for \$\\dot{X}_t\$
- `data`: Matrix (M, d) for \$X_1\$
- `noise`: Matrix (M, d) for \$X_0\$
- `interp::LinearInterpolation`
- `time::Number`
"""
function state_and_velocity!(state::M, velocity::M,
    data::M, noise::M, interp::LinearInterpolant, time::T) where {T,M<:AbstractMatrix{T}}
    M_samples, x_dim = size(data)
    @argcheck size(noise) == (M_samples, x_dim) DimensionMismatch
    @argcheck size(state) == (M_samples, x_dim) DimensionMismatch
    @argcheck size(velocity) == (M_samples, x_dim) DimensionMismatch
    alpha, beta = interp.alpha(time), interp.beta(time)
    alpha_dot, beta_dot = interp.alpha_deriv(time), interp.beta_deriv(time)
    @inbounds for idx in eachindex(data)
        state[idx] = alpha * data[idx] + beta * noise[idx]
        velocity[idx] = alpha_dot * data[idx] + beta_dot * noise[idx]
    end
    nothing
end

"""
	initialize_interpolant_spaces(data::AbstractMatrix{T}, fmset::FixedMultiIndexSet{dim_x_t}, rng)
Create spaces for storing intermediate results for learning the stochastic interpolant.
"""
function initialize_interpolant_spaces(data::AbstractMatrix{T},
    fmset::FixedMultiIndexSet{x_t_dim}, rng::AbstractRNG) where {T,x_t_dim}
    N_samples, x_dim = size(data)
    num_coeff = length(fmset)
    @argcheck x_dim == x_t_dim - 1 DimensionMismatch
    state = similar(data)
    velocity = similar(data)
    noise = similar(data)
    randn!(rng, noise)
    full_eval_space = similar(data, (num_coeff, N_samples))
    univariate_eval_space = ntuple(j -> similar(data, (fmset.max_orders[j] + 1, N_samples)), x_t_dim)
    LS_matrix = zeros(T, num_coeff, num_coeff)
    LS_vector = zeros(T, num_coeff, x_dim)
    state, velocity, noise, full_eval_space, univariate_eval_space, LS_matrix, LS_vector
end


"""
    create_LS_stoch_interp(data::AbstractMatrix, space_basis::MultivariateBasis, time_basis::UnivariateBasis,
    fmset::FixedMultiIndexSet, interp::AbstractStochInterpolant, time_pts, time_wts; [rng])
Create a least-squares system for the stochastic interpolant.

If `N=length(fmset)`, then this returns a `NxN` matrix and a `Nxd` vector, where
`d` is the dimension of the state (keeping in mind that fmset is d+1 dimensional due to time)

# Arguments
- `data` (M,d) samples drawn from target
- `basis` (d+1) basis for the space variables concatenated with time basis
- `fmset` (N,d+1) set of multi-indices
- `interp` stochastic interpolant
- `time_pts` (T) set of points to integrate loss over in time
- `time_wts` (T) set of weights for time integration rule. Defaults to uniform.
- `[rng]` (optional) RNG from Random
"""
function create_LS_stoch_interp(data::AbstractMatrix, space_basis::MultivariateBasis{x_dim}, time_basis::UnivariateBasis,
    fmset::FixedMultiIndexSet{x_t_dim}, interp::AbstractStochInterpolant,
    time_pts::AbstractVector, time_wts::AbstractVector=ones(length(time_pts));
    rng::AbstractRNG=Random.GLOBAL_RNG) where {x_dim, x_t_dim}

	@argcheck x_dim + 1 == x_t_dim DimensionMismatch
    @argcheck length(time_pts) == length(time_wts) DimensionMismatch
    N_samples = size(data, 1)
    spaces = initialize_interpolant_spaces(data, fmset, rng)
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