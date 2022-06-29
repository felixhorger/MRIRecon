
"""
	For the mask which requires permuted axes
	TODO: Make this a linear operator that sorts indices and then makes inplace modification, no dense array
	first element of shape is dynamic dimension, e.g. time
"""
function sampling_mask(shape::NTuple{N, Integer}, indices::AbstractVector{<: NTuple{M, Integer}}) where {N, M}
	@assert N == M + 1
	b = zeros(Float64, shape)
	n = shape[1]
	for (i, j) in enumerate(indices)
		b[mod1(i, n), j...] = 1
	end
	return b
end

"""
	For kspace data
	readout direction and channels must be first axes of a
"""
function sparse2dense(
	a::AbstractArray{<: Number, 3},
	shape::NTuple{N, Integer},
	indices::AbstractVector{<: NTuple{M, Integer}},
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

