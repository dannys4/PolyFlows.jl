export PolyFlowBasis, SparsePolyFlowBasis

abstract type AbstractPolyFlowBasis end

struct PolyFlowBasis{MVBasis<:MultivariateBasis, UBasis<:UnivariateBasis, FMset<:FixedMultiIndexSet} <: AbstractPolyFlowBasis
    space_basis::MVBasis
    time_basis::UBasis
    fmset::FMset
    function PolyFlowBasis(space_basis::_MV, time_basis::_UB, fmset::_FM) where {x_dim, x_t_dim, _MV<:MultivariateBasis{x_dim}, _UB, _FM<:FixedMultiIndexSet{x_t_dim}}
        @argcheck x_dim + 1 == x_t_dim DimensionMismatch
        new{_MV,_UB, _FM}(space_basis, time_basis, fmset)
    end
end

function Base.size(basis::PolyFlowBasis{T}) where {d,T<:MultivariateBasis{d}}
    (d,length(basis.fmset))
end

function Base.size(basis::PolyFlowBasis{T},idx::Int) where {d,T<:MultivariateBasis{d}}
    size(basis)[idx]
end


"""
    SparsePolyFlowBasis(space_bases, time_basis, sparse_selection, out_idxs, fmsets)
"""
struct SparsePolyFlowBasis{MVBases<:AbstractVector{<:MultivariateBasis},UBasis<:UnivariateBasis,BMat<:Union{Nothing,Matrix{Bool}},VIdx<:AbstractVector{<:Integer},VFmset<:AbstractVector{<:FixedMultiIndexSet}} <: AbstractPolyFlowBasis
    space_basis::MVBases
    time_basis::UBasis
    sparse_selection::BMat
    out_idxs::VIdx
    fmsets::VFmset
    function SparsePolyFlowBasis(space_bases::_MB, time_basis::_UB, sparse_selection::_BM, out_idxs::_VI, fmsets::_VF) where {_MB, _UB, _BM, _VI, _VF}
        d_in, d_out = size(sparse_selection)
        @argcheck d_in >= d_out
        # Need out_idxs to be vector of unique spatial coordinates corresponding to which output the fmset is w.r.t.
        @argcheck length(out_idxs) == d_out DimensionMismatch
        @argcheck length(unique(out_idxs)) == d_out
        # Sparse bases: num_out length vec of bases
        @argcheck length(space_bases) == d_out DimensionMismatch
        for out_idx in 1:d
            @argcheck out_idxs[out_idx] > 0 && out_idxs[out_idx] <= d_out
            num_varying = sum(sparse_selection[:,out_idx])
            @argcheck length(space_bases[out_idx]) == num_varying
            # Account for time in multi-index set
            @argcheck size(fmsets[out_idx],1) == num_varying + 1
        end
        new{_MB,_UB,_BM,_VI,_VF}(space_bases, time_basis, sparse_selection, out_idxs, fmsets)
    end
end


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
	initialize_interpolant_spaces(ScalarType, N_samples, fmset::FixedMultiIndexSet{dim_x_t}, rng)
Create spaces for storing intermediate results for learning the stochastic interpolant when using same mset for all dimensions
"""
function initialize_interpolant_spaces(::Type{T}, N_samples::Int,
    fmset::FixedMultiIndexSet{x_t_dim}, rng::AbstractRNG, full_out::Bool) where {T,x_t_dim}
    x_dim = x_t_dim-1
    num_coeff = length(fmset)
    @argcheck x_dim == x_t_dim - 1 DimensionMismatch
    state = Matrix{T}(undef, N_samples, x_dim)
    velocity = full_out ? Matrix{T}(undef, N_samples, x_dim) : Vector{T}(undef, N_samples)
    noise = randn(rng, T, N_samples, x_dim)
    full_eval_space = Matrix{T}(undef, num_coeff, N_samples)
    univariate_eval_space = ntuple(j -> Matrix{T}(undef, fmset.max_orders[j] + 1, N_samples), x_t_dim)
    LS_matrix = zeros(T, num_coeff, num_coeff)
    LS_vector = full_out ? zeros(T, num_coeff, x_dim) : zeros(T, num_coeff)
    state, velocity, noise, full_eval_space, univariate_eval_space, LS_matrix, LS_vector
end


initialize_interpolant_spaces(T, N_samples, basis::PolyFlowBasis, rng, full_out) = initialize_interpolant_spaces(T, N_samples, basis.fmset, rng, full_out)

velocity_basis_eval_step!(full_eval_space, univariate_eval_space, basis::PolyFlowBasis, state, t) = velocity_basis_eval_step!(full_eval_space, univariate_eval_space, basis.space_basis, basis.time_basis, basis.fmset, state, t)
