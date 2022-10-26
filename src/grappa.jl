
function grappa_neighbours(kernelsize::NTuple{N, Integer}, acceleration::NTuple{N, Integer}) where N
	reduced_size = kernelsize .÷ acceleration .+ mod.(kernelsize, acceleration)
	indices = Vector{CartesianIndex{N}}(undef, prod(reduced_size))
	for (i, I) in enumerate(CartesianIndices(reduced_size))
		indices[i] = CartesianIndex((Tuple(I) .- 1) .* acceleration .+ 1)
	end
	return indices
end

function grappa_kernel(calibration::AbstractArray{T, N}, neighbours::AbstractVector{<: CartesianIndex{D}}, kernelsize::NTuple{D, Integer}) where {T<:Number, N, D}
	# TODO check kernelsize
	num_channels = size(calibration, 1)
	shape = size(calibration)[2:N]
	# Get Hankel
	hankel = MRIHankel.hankel_matrix(calibration, neighbours, kernelsize)
	# Get subregion of calibration used to compute kernel
	offset = kernelsize .÷ 2 .+ 1
	indices = ntuple(d -> offset[d]:shape[d]-kernelsize[d]+offset[d]-1, Val(D))
	calibration = reshape(calibration[:, indices...], num_channels, prod(length.(indices)))
	g = calibration * pinv(hankel)
	g = reshape(g, num_channels, num_channels, length(neighbours))
	return g
end



"""
kspace needs to be zero at `indices`
"""
function apply_grappa_kernel!(
	kspace::AbstractArray{C, N},
	g::AbstractArray{C, 3},
	neighbours::AbstractVector{<: CartesianIndex{D}},
	kernelsize::NTuple{D, Integer},
	indices::AbstractVector{<: CartesianIndex{M}}
) where {C <: Complex, N, D, M}
	@assert N > 1
	@assert N == D + 1 # Channels
	@assert N == M + 2 # Readout and channels
	@assert MRIHankel.check_kernelsize(neighbours, kernelsize)

	# Get dimensions
	shape = size(kspace)
	num_channels = shape[1]
	spatial_shape = shape[2:N]
	@assert size(g) == (num_channels, num_channels, length(neighbours))

	# Subtract offset
	neighbours = [L - CartesianIndex(kernelsize .÷ 2 .+ 1) for L in neighbours]

	kspace_d = reinterpret(reshape, Float64, kspace)
	g_d = reinterpret(reshape, Float64, g)

	num_readout = shape[2]
	Threads.@threads for l in eachindex(neighbours)
		@inbounds L = neighbours[l]
		for K_phase in indices
			for k = 1:num_readout
				K = CartesianIndex(k, Tuple(K_phase)...)
				J = CartesianIndex(mod1.(Tuple(L + K), spatial_shape))
				@turbo for c_in = 1:num_channels, c_out = 1:num_channels
					kspace_d[1, c_out, K] += (
							g_d[1, c_out, c_in, l] * kspace_d[1, c_in, J]
						-	g_d[2, c_out, c_in, l] * kspace_d[2, c_in, J]
					)
					kspace_d[2, c_out, K] += (
							g_d[1, c_out, c_in, l] * kspace_d[2, c_in, J]
						+	g_d[2, c_out, c_in, l] * kspace_d[1, c_in, J]
					)
				end
			end
		end
	end
	return kspace
end

