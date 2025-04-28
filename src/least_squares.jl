export create_LS_stoch_interp

"""
	velocity_basis_eval_step!(full_eval_space, univariate_eval_space, basis, fmset, state)
Evaluate \$\\{\\Phi(X_t^{(j)},t)\\}_{j=1}^M\\subset \\mathbb{R}^{N}\$ for fixed \$t\$. Note that `state` should be \$\\{(X_t^{(j)},t)\\}\$, i.e., augmented with the time pt, so dimension is d+1

# Arguments
- `full_eval_space`: Matrix (N,M)
- `univariate_eval_space::NTuple{d+1}`: Each matrix is `(p_j,M)`, where `p_j` is the maximum degree of basis function in dim `j`
- `basis::MultivariateBasis{d+1}`
- `fmset::FixedMultiIndex{d+1}`
- `state` Matrix (M,d+1)
"""
function velocity_basis_eval_step!(full_eval_space::M, univariate_eval_space::NTuple{x_t_dim,M},
    basis::MultivariateBasis{x_t_dim}, fmset::FixedMultiIndexSet{x_t_dim},
    state::M) where {x_t_dim,T<:Number,M<:AbstractMatrix{T}}

    N_samples = size(state, 1)
    num_coeff = length(fmset)
    # Dimension checking
    @argcheck size(state, 2) == x_t_dim DimensionMismatch
    @argcheck size(full_eval_space) == (num_coeff, N_samples) DimensionMismatch
    for space in univariate_eval_space
        @argcheck(size(space, 2) == N_samples, DimensionMismatch)
    end

    # Evaluate the univariate basis at the data points
    Evaluate!(univariate_eval_space, basis, state)
    # Evaluate the full basis at the data points
    Evaluate!(full_eval_space, fmset, univariate_eval_space)
    nothing
end

"""
	state_and_velocity!(state, velocity, data, noise, interp, time)
Get \$X_t = \\alpha(t) X_1 + \\beta(t) X_0\$ as well as \$\\dot{X}_t\$, where \$X_1\$ is drawn from the target and \$X_0\$ from the reference.

Output results into `state` and `velocity`.

# Arguments
- `state`: Matrix (M, d+1) for \$(X_t, t)\$, i.e., augmenting the state with the time
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
    @argcheck size(state) == (M_samples, x_dim + 1) DimensionMismatch
    @argcheck size(velocity) == (M_samples, x_dim) DimensionMismatch
    alpha, beta = interp.alpha(time), interp.beta(time)
    alpha_dot, beta_dot = interp.alpha_deriv(time), interp.beta_deriv(time)
    @inbounds for idx in CartesianIndices(data)
        state[idx] = alpha * data[idx] + beta * noise[idx]
        velocity[idx] = alpha_dot * data[idx] + beta_dot * noise[idx]
    end
    state[:, end] .= time
    nothing
end

"""
	initialize_interpolant_spaces(data::AbstractMatrix{T}, fmset::FixedMultiIndexSet{dim_x_t}, rng)
Create spaces for storing intermediate results for learning the stochastic interpolant.
"""
function initialize_interpolant_spaces(data::AbstractMatrix{T},
    fmset::FixedMultiIndexSet{dim_x_t}, rng::AbstractRNG) where {T,dim_x_t}
    N_samples, x_dim = size(data)
    num_coeff = length(fmset)
    @argcheck x_dim == dim_x_t - 1 DimensionMismatch
    state = similar(data, (N_samples, x_dim + 1))
    velocity = similar(data)
    noise = similar(data)
    randn!(rng, noise)
    full_eval_space = similar(data, (num_coeff, N_samples))
    univariate_eval_space = ntuple(j -> similar(data, (fmset.max_orders[j] + 1, N_samples)), dim_x_t)
    LS_matrix = zeros(T, num_coeff, num_coeff)
    LS_vector = zeros(T, num_coeff, x_dim)
    state, velocity, noise, full_eval_space, univariate_eval_space, LS_matrix, LS_vector
end


"""
    create_LS_stoch_interp(data::AbstractMatrix, basis::MultivariateBasis, fmset::FixedMultiIndexSet, 
    interp::AbstractStochInterpolant, time_pts, time_wts; [rng])
Create a least-squares system for the stochastic interpolant.

If `N=length(fmset)`, then this returns a `NxN` matrix and a `Nxd` vector, where
`d` is the dimension of the state (keeping in mind that fmset is d+1 dimensional)

# Arguments
- `data` (M,d) samples drawn from target
- `basis` (d+1) basis for the space variables concatenated with time basis
- `fmset` (N,d+1) set of multi-indices
- `interp` stochastic interpolant
- `time_pts` (T) set of points to integrate loss over in time
- `time_wts` (T) set of weights for time integration rule. Defaults to uniform.
- `[rng]` (optional) RNG from Random
"""
function create_LS_stoch_interp(data::AbstractMatrix, basis::MultivariateBasis{x_t_dim},
    fmset::FixedMultiIndexSet{x_t_dim}, interp::AbstractStochInterpolant,
    time_pts::AbstractVector, time_wts::AbstractVector=ones(length(time_pts));
    rng::AbstractRNG=Random.GLOBAL_RNG) where {x_t_dim}

    @argcheck length(time_pts) == length(time_wts)
    N_samples = size(data, 1)
    spaces = initialize_interpolant_spaces(data, fmset, rng)
    state, velocity, noise, full_eval_space, univariate_eval_space, LS_matrix, LS_vector = spaces
    for time_idx in eachindex(time_pts)
        t, t_wt = time_pts[time_idx], time_wts[time_idx]
        # Create the state and velocity vectors at this time
        state_and_velocity!(state, velocity, data, noise, interp, t)
        # Evaluate the basis for this state
        velocity_basis_eval_step!(full_eval_space, univariate_eval_space, basis, fmset, state)
        # Add to the current least-squares matrix and vector
        mul!(LS_matrix, full_eval_space, full_eval_space', t_wt / N_samples, true)
        mul!(LS_vector, full_eval_space, velocity, t_wt / N_samples, true)
    end
    LS_matrix, LS_vector
end