
function grappa_kernel(calibration::AbstractArray{T, N}, neighbours::NTuple{M, CartesianIndex{D}}, kernelsize::NTuple{D, Integer}) where {T<:Number, N, M, D}
	# TODO check kernelsize
	num_channels = size(calibration, N)
	# Get Hankel
	hankel = MRIHankel.hankel_matrix(calibration, neighbours, kernelsize)
	hankel = reshape(hankel, size(hankel, 1), num_channels * M)
	# Get subregion of calibration used to compute kernel
	shape = size(calibration)[1:D]
	offset = kernelsize .÷ 2 .+ 1
	reduced_shape = shape .- kernelsize
	indices = ntuple(d -> offset[d]:shape[d]-offset[d]-mod(kernelsize[d], 2)+1, Val(D))
	calibration = reshape(calibration[indices..., :], prod(reduced_shape), num_channels)
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

	offset = CartesianIndex((kernelsize .÷ 2 .+ 1)...)
	i = 1 # counter for neighbours and num_channels
	for L in neighbours
		for K in indices
			for r = 1:shape[1] # Readout
				λ = CartesianIndex(r, Tuple(K)...)
				κ = L + λ - offset
				k = Tuple(κ)
				if !checkbounds(Bool, kspace, k..., :)
					k = mod1.(k, spatial_shape)
				end
				κ = CartesianIndex(k)
				@views kspace[λ, :] .+= transpose(transpose(kspace[κ, :]) * g[i:i+num_channels-1, :])
			end
		end
		i += num_channels # Next neighbour
	end
	return kspace
end

