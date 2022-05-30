
"""
	For the mask which requires permuted axes
	TODO: Make this a linear operator that sorts indices and then makes inplace modification, no dense array
"""
function sampling_mask(timepoints::Integer, indices::AbstractVector{<: Integer}...)
	b = zeros(Float64, timepoints, maximum.(indices)...)
	for (i, j) in enumerate(zip(indices...))
		b[mod1(i, timepoints), j...] = 1
	end
	return b
end

"""
	For kspace data
	readout direction and channels must be first axis of a
"""
function sparse2dense(a::AbstractArray{<: Number, 3}, timepoints::Integer, indices::AbstractVector{<: Integer}...)
	@assert all(size(a, 3) .== length.(indices))
	b = zeros(eltype(a), size(a, 1), maximum.(indices)..., size(a, 2), timepoints)
	for (i, j) in enumerate(zip(indices...))
		b[:, j..., :, mod1(i, timepoints)] = a[:, :, i]
	end
	return b
end

