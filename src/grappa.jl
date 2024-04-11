
# TODO: are readouts normally included? Disentangled?
function grappa_neighbours(kernelsize::NTuple{N, Integer}, acceleration::NTuple{N, Integer}) where N
	reduced_size = kernelsize .÷ acceleration .+ mod.(kernelsize, acceleration)
	return Tuple(
		CartesianIndex((Tuple(I) .- 1) .* acceleration .+ 1)
		for (i, I) in enumerate(CartesianIndices(reduced_size))
	)
end

# TODO: is tuple or array here better?
function grappa_kernel(calibration::AbstractArray{T, N}, targets::Tuple{Vararg{CartesianIndex{D}}}, neighbours::Tuple{Vararg{CartesianIndex{D}}}, kernelsize::NTuple{D, Integer}) where {T<:Number, N, D}
	# TODO checks for shape and kernel size
	num_channels = size(calibration, 1)
	shape = size(calibration)[2:N]
	# Get Hankel
	hankel = MRIHankel.hankel_matrix(calibration, neighbours, kernelsize)
	# Get target values
	if length(targets) == 1
		@assert MRIHankel.check_kernelsize(targets, kernelsize)
		# Single target, simple copy is enough
		target = targets[1]
		indices = ntuple(d -> target[d]:shape[d]-kernelsize[d]+target[d]-1, Val(D))
		calibration = reshape(calibration[:, indices...], num_channels, prod(length.(indices)))
	else
		# Multiple targets, need to construct Hankel matrix
		calibration = MRIHankel.hankel_matrix(calibration, targets, kernelsize)
		# This has size (num_channels * num_targets, number of instances of the kernel fitting into calibration)
		#@show size(calibration)
	end
	g = calibration * pinv(hankel)
	g = reshape(g, num_channels, length(targets), num_channels, length(neighbours))
	g = permutedims(g, (1, 3, 2, 4))
	return g
end




# TODO: how to deal with cases where threading should be disabled, e.g. for MRF, parallelisation should happen along time not in k-space?
"""
kspace needs to be zero where not sampled
indices is where you want the kernel applied, location of centre of kernel
Wrap around is used, i.e. the kernel's "unit cell" must fit an integer number of times into the k-space
"""
function apply_grappa_kernel!(
	kspace::AbstractArray{C, N},
	g::AbstractArray{C, 4},
	targets::Tuple{Vararg{CartesianIndex{D}}},
	neighbours::Tuple{Vararg{CartesianIndex{D}}},
	kernelsize::NTuple{D, Integer},
	indices::AbstractVector{<: CartesianIndex{M}}
) where {C <: Complex, N, D, M}
	@assert N > 1
	@assert N == D + 1 # Channels
	@assert N == M + 2 # Readout and channels TODO make proper error msg
	@assert all(MRIHankel.check_kernelsize(i, kernelsize) for i in (targets, neighbours))

	# Get dimensions
	shape = size(kspace)
	num_channels = shape[1]
	spatial_shape = shape[2:N]
	@assert size(g) == (num_channels, num_channels, length(targets), length(neighbours))

	kspace_d = reinterpret(reshape, Float64, kspace)
	g_d = reinterpret(reshape, Float64, g)

	one_shift(Is) = ntuple(i -> Is[i] - one(CartesianIndex{D}), length(Is))
	targets, neighbours = one_shift.((targets, neighbours))
	num_readout = shape[2]

	wrap_index(I) = CartesianIndex(mod1.(Tuple(I), spatial_shape))
	offset = CartesianIndex(kernelsize .÷ 2)
	Threads.@threads for K_phase in indices
		for l in eachindex(neighbours)
			@inbounds L = neighbours[l]
			for t in eachindex(targets)
				@inbounds T = targets[t]
				for k = 1:num_readout
					K = CartesianIndex(k, Tuple(K_phase)...)
					K = K - offset
					X, Y = wrap_index.((K + L, K + T))
					#@turbo
					for c_in = 1:num_channels, c_out = 1:num_channels
						kspace_d[1, c_out, Y] += (
							  g_d[1, c_out, c_in, t, l] * kspace_d[1, c_in, X]
							- g_d[2, c_out, c_in, t, l] * kspace_d[2, c_in, X]
						)
						kspace_d[2, c_out, Y] += (
							  g_d[1, c_out, c_in, t, l] * kspace_d[2, c_in, X]
							+ g_d[2, c_out, c_in, t, l] * kspace_d[1, c_in, X]
						)
					end
				end
			end
		end
	end
	return kspace
end

