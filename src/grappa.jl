
function grappa_kernel(calibration::AbstractArray{T, N}, neighbours::NTuple{M, CartesianIndex{D}}) where {T<:Number, N, M, D}
	num_channels = size(calibration, N)
	# Get Hankel
	kernelsize = Tuple(maximum(I[i] for I in neighbours) for i = 1:D)
	hankel = MRIHankel.hankel_matrix(calibration, kernelsize, neighbours)
	hankel = reshape(hankel, size(hankel, 1), num_channels * M)
	# Get subregion of calibration used to compute kernel
	shape = size(calibration)[1:D]
	reduced_shape = shape .- kernelsize
	indices = centre_indices.(shape, reduced_shape)
	calibration = reshape(calibration[indices..., :], prod(reduced_shape), num_channels)
	# Compute GRAPPA kernel channelwise (otherwise hankel would need to be repeated)
	g = pinv(hankel) * calibration
	return g
end

