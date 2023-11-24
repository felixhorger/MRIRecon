# TODO: rename sampling?

"""
	dense sampling mask
	TODO: remove the leading one-element dim. It's annoying
"""
function sampling_mask(
	indices::AbstractVector{<: CartesianIndex{N}},
	shape::NTuple{N, Integer},
	num_dynamic::Integer
) where N
	b = zeros(Bool, 1, shape..., num_dynamic) # The one is for the readout axis
	for (i, j) in enumerate(indices)
		b[1, j, mod1(i, num_dynamic)] = true
	end
	return b
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
	num_dynamic::Integer;
	do_sort::Bool=true
) where N
	min_indices_per_dynamic = length(indices) ÷ num_dynamic # TODO: rename remove min
	split_indices = Vector{Vector{CartesianIndex{N}}}(undef, num_dynamic)
	for j = 1:num_dynamic
		ind = Vector{CartesianIndex{N}}(undef, 0)
		sizehint!(ind, min_indices_per_dynamic + (j > mod(length(indices), num_dynamic) ? 0 : 1))
		split_indices[j] = ind
	end
	if do_sort
		for i in eachindex(indices)
			index = indices[i]
			j = mod1(i, num_dynamic)
			indices_of_dynamic = split_indices[j]
			k = searchsortedfirst(indices_of_dynamic, index)
			insert!(indices_of_dynamic, k, index)
		end
	else # TODO: this isn't nice coding
		for i in eachindex(indices)
			index = indices[i]
			j = mod1(i, num_dynamic)
			indices_of_dynamic = split_indices[j]
			push!(indices_of_dynamic, index)
		end
	end
	return split_indices
end

function split_sampling_spatially(
	indices::AbstractVector{<: CartesianIndex{N}},
	shape::NTuple{N, Integer},
	num_dynamic::Integer
) where N
	split_indices_spatially = [Vector{Int}(undef, 0) for _ in CartesianIndices(shape)]
	for i in eachindex(indices)
		j = mod1(i, num_dynamic)
		I = indices[i]
		#k = searchsortedfirst(split_indices_spatially[I], j)
		#insert!(split_indices_spatially[I], k, j)
		push!(split_indices_spatially[I], j)
	end
	return split_indices_spatially
end
function split_sampling(split_indices_spatially::AbstractArray{<: AbstractVector{<: Integer}, N}, num_dynamic::Integer) where N
	split_indices = [Vector{CartesianIndex{N}}(undef, 0) for _ = 1:num_dynamic]
	for I in CartesianIndices(split_indices_spatially)
		for t in split_indices_spatially[I]
			push!(split_indices[t], I)
		end
	end
	return split_indices
end

"""
"""
function in_chronological_order(split_indices::AbstractVector{<: AbstractVector{<: CartesianIndex{N}}}) where N
	indices = Vector{CartesianIndex{N}}(undef, sum(length.(split_indices)))
	num_dynamic = length(split_indices)
	for i in eachindex(indices)
		k, j = divrem(i-1, num_dynamic) .+ 1
		indices[i] = split_indices[j][k]
	end
	return indices
end
function in_chronological_order(
	split_indices_spatially::AbstractArray{<: AbstractVector{<: Integer}, N},
	num_dynamic::Integer
) where N
	return in_chronological_order(
		split_sampling(split_indices_spatially, num_dynamic)
	)
end

"""
	For dynamic
"""
function unsampled_indices(
	indices::AbstractVector{<: CartesianIndex{N}},
	shape::NTuple{N, Integer},
	num_dynamic::Integer
) where N
	# Split into dynamic frames
	split_indices = split_sampling(indices, num_dynamic)
	# Sort within each dynamic frame and find unsampled indices
	indices_to_mask = Vector{Vector{CartesianIndex{N}}}(undef, num_dynamic)
	@views for j = 1:num_dynamic
		indices_to_mask[j] = unsampled_indices(split_indices[j], shape)
	end
	return indices_to_mask
end



function is_unique_sampling(indices::AbstractVector{<: CartesianIndex{N}}, num_dynamic::Integer) where N
	split_indices = split_sampling(indices, num_dynamic)
	for i = 1:num_dynamic
		allunique(split_indices[i]) && continue
		return false
	end
	return true
end

"""
	TODO: is this useful?
"""
function find_duplicates(indices::AbstractVector{<: CartesianIndex{N}}, num_dynamic::Integer) where N
	duplicates = Vector{Int}(undef, 0)
	sizehint!(duplicates, length(indices) ÷ 2)
	j = zero(CartesianIndex{N})
	for i in eachindex(indices)
		k = indices[i]
		if j == k
			push!(duplicates, i)
		end
		j = k
	end
	return duplicates
end

"""
	a[spatial dims..., channels, dynamic]

	no inbounds checks!
"""
function apply_sparse_mask!(a::Array{<: Number, N}, indices_to_mask::Vector{Vector{CartesianIndex{D}}}) where {N, D}
	@assert N == D + 3 # Readout, channels, dynamic
	for j = 1:size(a, N)
		# TODO: Maybe threads?
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
	shape = (readout, phase encode ..., channels, num_dynamic)
"""
function plan_masking!(
	indices::AbstractVector{<: CartesianIndex{N}},
	shape::NTuple{M, Integer};
	dtype::Type=ComplexF64 # Needed for type stability
) where {N, M}
	@assert N == M - 3 "indices must include only phase encoding, shape = (readout, phase encode..., channels, dynamic)"
	# Readout, channels, dynamic

	# Get unsampled indices
	indices_to_mask = unsampled_indices(indices, shape[2:N+1], shape[M])

	# Construct linear operator
	U = HermitianOperator{dtype}(
		prod(shape),
		(y, x) -> begin
			@assert length(y) == 0
			x_in_shape = reshape(x, shape)
			apply_sparse_mask!(x_in_shape, indices_to_mask)
			x
		end
	)

	return U
end

function plan_regular_undersampling!(
	shape::NTuple{D, Integer},
	dim::Integer,
	r::Integer,
	offset::Integer;
	dtype::Type=ComplexF64 # Needed for type stability
) where D
	num_full = prod(shape)
	num_dim = shape[dim]
	stride_dim = prod(shape[1:dim-1])
	U = HermitianOperator{dtype}(
		prod(shape),
		(y, x) -> begin
			# length of x is checked by LinearOperators.jl
			@assert length(y) == 0
			Threads.@threads for i = 1:num_full
				# Get index of dim
				j = mod1((i-1) ÷ stride_dim + 1, num_dim)
				mod1(j, r) == offset && continue
				@inbounds x[i] = 0
			end
			x
		end
	)
	return U
end


"""
l linear index of full array
i_dim index of dimension in full array == i_dim = mod1((i-1) ÷ stride_dim + 1, num_dim)
step slicing step
offset first element, i.e. offset:step:end
"""
function sliced_index(l::Integer, step::Integer, offset::Integer, num_dim::Integer, stride_dim::Integer, stride_after::Integer)
	l_dim_div, i_before = divrem(l-1, stride_dim)
	i_dim = mod(l_dim_div, num_dim) + 1
	mod1(i_dim, step) != offset && return 0
	i_after = l_dim_div ÷ num_dim
	return (
		1 + i_before
		+ ((i_dim - offset) ÷ step) * stride_dim
		+ i_after * stride_after
	)
end



regularly_undersampled_length(full::Integer, offset::Integer, r::Integer) = (full - offset) ÷ r + 1
function regular_undersampling_size(
	shape::NTuple{D, Integer},
	dim::Integer,
	r::Integer,
	offset::Integer
) where D
	@assert offset ≤ r
	@assert 0 < dim ≤ D
	num_full = prod(shape)
	num_dim_undersampled = regularly_undersampled_length(shape[dim], offset, r)
	num_undersampled = num_dim_undersampled * prod(shape[1:dim-1]) * prod(shape[dim+1:D])
	return num_undersampled, num_full
end
function plan_regular_undersampling(
	shape::NTuple{D, Integer},
	dim::Integer,
	r::Integer,
	offset::Integer;
	dtype::Type{T}=ComplexF64, # Needed for type stability
	Ux::AbstractVector{<: T}=empty(Vector{dtype}),
	UHy::AbstractVector{<: T}=empty(Vector{dtype}),
) where {D, T <: Number}
	@assert offset ≤ r
	@assert 0 < dim ≤ D
	@assert iseven(shape[dim]) "Odd dimension is not supported due to fftshift, not implemented"
	num_dim = shape[dim]
	stride_dim = prod(shape[1:dim-1])
	num_dim_undersampled = regularly_undersampled_length(shape[dim], offset, r)
	stride_after = stride_dim * num_dim_undersampled
	num_undersampled, num_full = regular_undersampling_size(shape, dim, r, offset)
	U = LinearOperator{dtype}(
		(num_undersampled, num_full),
		(y, x) -> begin
			# Lengths checked by LinearOperators.jl
			@inbounds Threads.@threads for i = 1:num_full
				k = sliced_index(i, r, offset, num_dim, stride_dim, stride_after)
				k == 0 && continue
				y[k] = x[i]
			end
			y
		end;
		adj = (x, y) -> begin
			# Lengths checked by LinearOperators.jl
			@inbounds Threads.@threads for i = 1:num_full
				k = sliced_index(i, r, offset, num_dim, stride_dim, stride_after)
				if k != 0
					x[i] = y[k]
				else
					x[i] = 0
				end
			end
			x
		end,
		out=check_allocate(Ux, num_undersampled),
		out_adj_inv=check_allocate(UHy, num_full)
	)
	return U
end

#= Instead of FT -> masking -> iFT, one can just sum up array elements (convolution)
Since sampling is regular, the convolution is trivial to compute.
This is not useful for what I need:

A = S' Fz' Fxy' M M_z Fxy Fz S  =  S' Fz' M_z Fz Fxy' M Fxy S

M_z is a low-rank mixing matrix with (1, 0, 1, 0...) along the diagonal.
half the memory could be saved by not "accumulating" Fz' M_z Fz.

For some other scenario it might be of use though, say full sampling along x and y
and r = 2 along partitions:
A = S' Fz' Fxy' Uz Fxy Fz S = S' Fz' Uz Fz S = S' Tz S

function plan_regular_undersampling_toeplitz()
end
=#



"""
	Use this if each k-space sample is acquired n times, where n > 1 and doesn't depend on k
	Convert CartesianIndex to Int first using linear indices: LinearIndices(shape)[cartesian_indices]
	Go through a, split indices into dynamics and if an index is already there,
	then add that k-space value to the one already registered and put zero into the duplicate.
	Also remember how many duplicates there are, and finally divide by those counts.
	This then enables to use the functions sparse2dense() and lowrank_sparse2dense (from MRFingerprinting.jl)
"""
function average_duplicates!(a::AbstractArray{<: Number, 3}, indices::AbstractVector{<: Integer}, num_dynamic::Integer)
	@assert size(a, 3) == length(indices)
	# Setup arrays
	min_indices_per_dynamic = length(indices) ÷ num_dynamic
	split_indices = Vector{Vector{Int}}(undef, num_dynamic) # Sampled indices split into dynamics
	split_indices_a = Vector{Vector{Int}}(undef, num_dynamic) # corresponding indices in a
	num_duplicates = Dict{Int, Int}() # index in a => number of samples
	# Setup sub-arrays and give size hints
	for j = 1:num_dynamic
		ind = Vector{Int}(undef, 0)
		ind_a = Vector{Int}(undef, 0)
		expected_length = min_indices_per_dynamic + (j > mod(length(indices), num_dynamic) ? 0 : 1)
		sizehint!.((ind, ind_a), expected_length)
		split_indices[j] = ind
		split_indices_a[j] = ind_a
	end
	# First samples go untouched
	for i = 1:num_dynamic
		push!(split_indices[i], indices[i])
		push!(split_indices_a[i], i)
	end
	# For all other samples need to check if already sampled at that dynamic
	for i = num_dynamic+1:length(indices)
		index = indices[i]
		j = mod1(i, num_dynamic)
		indices_of_dynamic = split_indices[j]
		k = searchsortedfirst(indices_of_dynamic, index)
		other_index = indices_of_dynamic[k]
		if other_index == index
			i_before = split_indices_a[j][k]
			@views begin
				a[:, :, i_before] .+= a[:, :, i]
				a[:, :, i] .= 0
			end
			if !haskey(num_duplicates, i_before)
				num_duplicates[i_before] = 2
			else
				num_duplicates[i_before] += 1
			end
		else
			insert!(indices_of_dynamic, k, index)
			insert!(split_indices_a[j], k, i)
		end
	end
	# Finish averaging by dividing by number of duplicates
	@views for (i, duplicates) in pairs(num_duplicates)
		a[:, :, i] ./= duplicates
	end
	return a
end



# TODO provide output type, maybe as sparse2dense{T}() if that's possible, think I saw that somewhere
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
	num_dynamic::Integer;
	t0::Integer=1
) where N
	@assert size(a, 3) == length(indices)
	perm = sortperm(indices)
	b = zeros(eltype(a), size(a, 1), size(a, 2), shape..., num_dynamic)
	# TODO: split indices, then iterate over that, also below
	t0 -= 1
	for i in eachindex(perm)
		j = perm[i]
		dynamic = mod1(j + t0, num_dynamic)
		k = indices[j]
		@views b[:, :, k, dynamic] .+= a[:, :, j]
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
	perm = sortperm(indices)
	offset = CartesianIndex(ntuple(d -> roi[d][1] - 1, N))
	b = zeros(eltype(a), size(a, 1), size(a, 2), roi_shape..., num_dynamic)
	for i in eachindex(perm)
		j = perm[i]
		k = indices[j]
		any(ntuple(d -> k[d] ∉ roi[d], N)) && continue
		dynamic = mod1(j, num_dynamic)
		@views b[:, :, k - offset, dynamic] .+= a[:, :, j]
	end
	return b
end


@inline centre_offset(i::Integer) = (i ÷ 2) + mod(i, 2)

# TODO: Maybe rewrite this to use CartesianIndices
function centre_indices(shape::Integer, centre_size::Integer)
	centre = shape ÷ 2 # Actually the centre is at this plus one
	half = centre_size ÷ 2
	lower = centre - half + 1
	upper = centre + half + mod(centre_size, 2)
	return lower:upper
end

