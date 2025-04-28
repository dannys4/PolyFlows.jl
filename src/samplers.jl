export deterministic_sample
using UnivariateApprox: UnivariateBasis

function deterministic_sample(space_basis::MultivariateBasis{x_dim}, time_basis::UnivariateBasis, fmset::FixedMultiIndexSet{x_t_dim}, integrator::AbstractSampleIntegrator, coeffs, N_samples, N_time=25; rng = Random.GLOBAL_RNG) where {x_dim,x_t_dim}
	@argcheck x_dim + 1 == x_t_dim DimensionMismatch
	@argcheck size(coeffs) == (length(fmset), x_dim) DimensionMismatch
	allUnivariateEvals = ntuple(j->Matrix{Float64}(undef, fmset.max_orders[j] + 1, N_samples), x_t_dim)
	spaceUnivariateEvals = ntuple(j->allUnivariateEvals[j], x_dim)
	timeUnivariateEvals = allUnivariateEvals[end]
	tmp_time = zeros(N_samples)
	sample_space = randn(rng, N_samples, x_dim)
	function sample_ode!(output, state, time)
		fill!(tmp_time, time)
		Evaluate!(spaceUnivariateEvals, space_basis, state)
		Evaluate!(timeUnivariateEvals, time_basis, tmp_time)
		for out_idx in 1:x_dim
			out_space = @view output[:, out_idx]
			coeff_out = @view coeffs[:, out_idx]
			basisAssembly!(out_space, fmset, coeff_out, allUnivariateEvals)
		end
	end
	for j in 1:N_time
		integrate!(integrator, sample_ode!, sample_space, (j-1)/N_time, 1/N_time)
		copy!(sample_space, get_output(integrator))
	end
	sample_space
end