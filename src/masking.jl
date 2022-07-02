
"""
	dense sampling mask
	last element of shape is dynamic dimension, e.g. time
"""
function sampling_mask(
	shape::NTuple{N, Integer},
	indices::AbstractVector{<: NTuple{D, Integer}}
) where {N, D}
	@assert N == D + 1
	b = zeros(Float64, shape)
	n = shape[1]
	for (i, j) in enumerate(indices)
		b[j..., mod1(i, n)] = 1
	end
	return b
end


"""
	Given indices which are sampled, find the ones which are not sampled.
	Spatial shape and indices must be sorted,
"""
@generated function unsampled_indices(
	shape::NTuple{N, Integer},
	indices::AbstractVector{<: NTuple{N, Integer}}
) where N
	return quote
		@assert length(indices) != 0
		indices_to_mask = Vector{NTuple{$N, Int}}(undef, prod(shape[1:$N]) - length(indices))
		i = 1 # Counter for indices
		j = 1 # Counter for indices_to_mask
		@nextract $N y (d -> indices[1][d]) # Get initial spatial index that is sampled
		@nloops $N x (d -> 1:shape[d]) begin
			if @nall $N (d -> x_d == y_d)
				# Get next spatial index which is sampled
				@nextract $N y (d -> indices[i][d])
				i += 1
			else
				# Index isn't sampled, store it
				indices_to_mask[j] = @ntuple $N (d -> x_d)
				j += 1
			end
		end
		return indices_to_mask
	end
end

"""
	a[spatial dims..., channels, dynamic]

"""
function apply_sparse_mask!(a::Array{<: Number, N}, indices_to_mask::Vector{Vector{NTuple{D, Int64}}}) where {N, D}
	@assert N == D + 2
	for j = 1:size(a, N)
		for i in indices_to_mask[j]
			a[i..., :, j] *= 0
		end
	end
	return a
end

"""
	Linear operator to perform masking efficiently (in place, sparse mask)
	last element in shape is other dims e.g. dynamic
	second last is channels
"""
function plan_masking(
	shape::NTuple{N, Integer},
	indices::AbstractVector{<: NTuple{D, Integer}}
) where {N, D}
	@assert N == D + 2

	spatial_shape = shape[1:D]
	# Note: channels = shape[N-1]
	num_dynamic = shape[N]

	# Get linear indices for the spatial dimensions
	linear_indices = LinearIndices(spatial_shape)

	# Split into frames
	split_indices = [
		Vector{NTuple{D, Int64}}(undef, (length(indices) ÷ num_dynamic) + 1)
		for _ in 1:num_dynamic
	]
	readouts_per_dynamic = zeros(Int64, num_dynamic)
	for i in eachindex(indices)
		j = mod1(i, num_dynamic)
		readouts_per_dynamic[j] += 1
		split_indices[j][readouts_per_dynamic[j]] = indices[i]
	end

	# Sort within each dynamic frame and find unsampled indices
	num_unsampled_indices = prod(shape[1:D]) - length(indices)
	indices_to_mask = Vector{Vector{NTuple{D, Int64}}}(undef, num_dynamic)
	let by = (t::NTuple{D, Integer} -> linear_indices[t...])
		@views for j = 1:num_dynamic
			a_dynamic = split_indices[j][1:readouts_per_dynamic[j]]
			sort!(a_dynamic; by)
			indices_to_mask[j] = unsampled_indices(spatial_shape, a_dynamic)
		end
	end

	# Construct linear operator
	MHM = LinearMap{ComplexF64}(
		x::AbstractVector{<: Complex} -> begin
			x_in_shape = reshape(x, shape)
			apply_sparse_mask!(x_in_shape, indices_to_mask)
			x
		end,
		prod(shape),
		ishermitian=true,
		issymmetric=true
	)

	return MHM
end


"""
	For kspace data
	readout direction and channels must be first axes of a
	dynamic dimension is the last, assumed mod(readout index, shape[N])
"""
function sparse2dense(
	a::AbstractArray{<: Number, 3},
	shape::NTuple{N, Integer},
	indices::AbstractVector{<: NTuple{M, Integer}}
) where {N, M}
	@assert N == M + 1
	@assert size(a, 3) == length(indices)
	n = shape[N]
	b = zeros(eltype(a), size(a, 1), shape[1:N-1]..., size(a, 2), n)
	for (i, j) in enumerate(indices)
		b[:, j..., :, mod1(i, n)] = a[:, :, i]
	end
	return b
end

