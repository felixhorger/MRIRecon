# TODO: rename sampling

"""
	dense sampling mask
"""
function sampling_mask(
	indices::AbstractVector{<: CartesianIndex{N}},
	shape::NTuple{N, Integer},
	num_dynamic::Integer
) where N
	b = zeros(Bool, 1, shape..., num_dynamic) # The one is for the readout axis
	for (i, j) in enumerate(indices)
		b[1, j, mod1(i, num_dynamic)] = 1
	end
	return b
end


import Base: sort
function sort(indices::AbstractVector{<: CartesianIndex{N}}, shape::NTuple{N, Integer}) where N
	linear_indices = LinearIndices(shape)
	sorted_indices = sort(indices; by=(k::CartesianIndex{N} -> linear_indices[k]))
	return sorted_indices
end



"""
	Given indices which are sampled, find the ones which are not sampled.
	indices must be sorted

	Indices outside shape are ignored!
"""
@generated function unsampled_indices(
	indices::AbstractVector{<: CartesianIndex{N}},
	shape::NTuple{N, Integer}
) where N
	return quote
		@assert length(indices) != 0
		indices_to_mask = Vector{CartesianIndex{$N}}(undef, prod(shape[1:$N]) - length(indices)÷shape[$N] - 1)
		i = 1 # Counter for indices
		j = 0 # Counter for indices_to_mask
		@nextract $N y (d -> indices[1][d]) # Get initial spatial index that is sampled
		@nloops $N x (d -> 1:shape[d]) begin
			if @nany $N (d -> x_d != y_d)
				# Index isn't sampled, store it
				j += 1
				indices_to_mask[j] = $(Expr(:call, :CartesianIndex, ntuple(d -> Symbol("x_$d"), Val(N))...))
				continue
			end
			while (@nall $N (d -> x_d == y_d)) && i < length(indices) # Correct because i is incremented before access
				# Get next spatial index which is sampled
				i += 1
				@nextract $N y (d -> indices[i][d])
			end
		end
		return indices_to_mask[1:j]
	end
end


"""
	One vector for each dynamic
"""
function split_sampling(
	indices::AbstractVector{<: CartesianIndex{N}},
	num_dynamic::Integer
) where N

	min_indices_per_dynamic = length(indices) ÷ num_dynamic
	split_indices = [
		Vector{CartesianIndex{N}}(
			undef,
			min_indices_per_dynamic + (j > mod(length(indices), num_dynamic) ? 0 : 1)
		)
		for j in 1:num_dynamic
	]
	split_i = zeros(Int64, num_dynamic)
	for i in eachindex(indices)
		j = mod1(i, num_dynamic)
		split_i[j] += 1
		split_indices[j][split_i[j]] = indices[i]
	end
	return split_indices
end

function is_unique_sampling(indices::AbstractVector{<: CartesianIndex{N}}, num_dynamic::Integer) where N
	split_indices = split_sampling(indices, num_dynamic)
	for i = 1:num_dynamic
		length(Set(split_indices[i])) != length(split_indices[i]) && return false
	end
	return true
end


"""
	a[spatial dims..., channels, dynamic]

	no inbounds checks!
"""
function apply_sparse_mask!(a::Array{<: Number, N}, indices_to_mask::Vector{Vector{CartesianIndex{D}}}) where {N, D}
	@assert N == D + 3 # Readout, channels, dynamic
	for j = 1:size(a, N)
		@inbounds for i in indices_to_mask[j]
			a[:, i, :, j] .= 0
		end
	end
	return a
end


@inline function valid_indices(indices::AbstractVector{<: CartesianIndex{N}}, shape::NTuple{N, Integer}) where N
	return all(v -> all(0 .< v .≤ shape), indices)
end


"""
	Linear operator to perform masking efficiently (in place, sparse mask)
	
	indices outside shape are ignored!
"""
function plan_masking(
	indices::AbstractVector{<: CartesianIndex{N}},
	target_shape::NTuple{M, Integer}
) where {N, M}
	@assert N == M - 3 # Readout, channels, dynamic

	# Get dimensions
	shape = target_shape[2:N+1]
	num_dynamic = target_shape[M]

	# Split into dynamic frames
	split_indices = split_sampling(indices, num_dynamic)

	# Sort within each dynamic frame and find unsampled indices
	indices_to_mask = Vector{Vector{CartesianIndex{N}}}(undef, num_dynamic)
	let
		linear_indices = LinearIndices(shape)
		by = (t::CartesianIndex{N} -> linear_indices[t])
		@views for j = 1:num_dynamic
			sort!(split_indices[j]; by)
			indices_to_mask[j] = unsampled_indices(split_indices[j], shape)
		end
	end

	# Construct linear operator
	U = HermitianOperator(
		prod(target_shape),
		x -> begin
			x_in_shape = reshape(x, target_shape)
			apply_sparse_mask!(x_in_shape, indices_to_mask)
			x
		end
	)

	return U
end


"""
	For kspace data
	readout direction and channels must be first axes of a
	dynamic dimension is the last, assumed mod1(readout index, shape[N])
	returns kspace[readout, spatial dims..., channels, dynamics]

	if num_dynamic not given than that dim is dropped
"""
function sparse2dense(
	a::AbstractArray{<: Number, 3},
	indices::AbstractVector{<: CartesianIndex{N}},
	shape::NTuple{N, Integer},
	num_dynamic::Integer
) where N
	@assert size(a, 3) == length(indices)
	linear_indices = LinearIndices(shape)
	perm = sortperm(indices; by=(k::CartesianIndex{N} -> linear_indices[k]))
	b = zeros(eltype(a), size(a, 1), size(a, 2), shape..., num_dynamic)
	for i in eachindex(perm)
		j = perm[i]
		dynamic = mod1(j, num_dynamic)
		k = indices[j]
		@views b[:, :, k, dynamic] = a[:, :, j]
	end
	return b
end

function sparse2dense_trunc(
	a::AbstractArray{<: Number, 3},
	indices::AbstractVector{<: CartesianIndex{N}},
	shape::NTuple{N, Integer},
	roi::NTuple{N, UnitRange{<: Integer}},
	roi_shape::NTuple{N, Integer},
	num_dynamic::Integer
) where N
	@assert size(a, 3) == length(indices)
	linear_indices = LinearIndices(shape)
	perm = sortperm(indices; by=(k::CartesianIndex{N} -> linear_indices[k]))
	offset = CartesianIndex(ntuple(d -> roi[d][1] - 1, N))
	b = zeros(eltype(a), size(a, 1), size(a, 2), roi_shape..., num_dynamic)
	for i in eachindex(perm)
		j = perm[i]
		k = indices[j]
		any(ntuple(d -> k[d] ∉ roi[d], N)) && continue
		dynamic = mod1(j, num_dynamic)
		@views b[:, :, k - offset, dynamic] = a[:, :, j]
	end
	return b
end


# TODO: Maybe rewrite this to use CartesianIndices
function centre_indices(shape::Integer, centre_size::Integer)
	centre = shape ÷ 2
	half = centre_size ÷ 2
	lower = centre - half + 1
	upper = centre + half + mod(centre_size, 2)
	return lower:upper
end

