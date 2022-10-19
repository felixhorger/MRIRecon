
function grappa_kernel(calibration::AbstractArray{T, N}, neighbours::NTuple{M, CartesianIndex{D}}, kernelsize::NTuple{D, Integer}) where {T<:Number, N, M, D}
	# TODO check kernelsize
	shape = size(calibration)[1:D]
	num_channels = size(calibration, N)
	# Get Hankel
	hankel = MRIHankel.hankel_matrix(calibration, neighbours, kernelsize)
	hankel = reshape(hankel, size(hankel, 1), num_channels * M)
	# Get subregion of calibration used to compute kernel
	offset = kernelsize .÷ 2 .+ 1
	indices = ntuple(d -> offset[d]:shape[d]-kernelsize[d]+offset[d]-1, Val(D))
	calibration = reshape(calibration[indices..., :], prod(length.(indices)), num_channels)
	g = pinv(hankel) * calibration
	return g
end

"""
kspace needs to be zero at `indices`
"""
function apply_grappa_kernel!(
	kspace::AbstractArray{T, N},
	g::AbstractMatrix{T},
	neighbours::NTuple{M, CartesianIndex{D}},
	kernelsize::NTuple{D, Integer},
	indices::AbstractVector{<: CartesianIndex{E}}
) where {T <: Number, N, M, D, E}
	@assert N > 1
	@assert N == D + 1 # Channels
	@assert N == E + 2 # Readout and channels
	@assert MRIHankel.check_kernelsize(neighbours, kernelsize)

	# Get dimensions
	shape = size(kspace)
	num_channels = shape[N]
	spatial_shape = shape[1:D]
	num_neighbours_and_channels = M * num_channels
	@assert size(g) == (num_neighbours_and_channels, num_channels)

	# Make kspace a circular array
	circular_kspace = CircularArray(kspace)

	offset = CartesianIndex((kernelsize .÷ 2 .+ 1)...)
	for c_out = 1:num_channels
		for c_in = 1:num_channels
			i = 1 # counter for neighbours and num_channels
			for L in neighbours
				@inbounds γ = g[i+c_in-1, c_out]
				for K in indices
					@simd for r = 1:shape[1] # Readout
						λ = CartesianIndex(r, Tuple(K)...)
						κ = L + λ - offset
						@inbounds kspace[λ, c_out] += γ * circular_kspace[κ, c_in]
					end
				end
				i += num_channels # Next neighbour
			end
		end
	end
	return kspace
end

