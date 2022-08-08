# Uecker2019 ENLIVE
# Uecker2017 ESPIRIT+Virtual Conjugate Coils
# Iyer2020 Sure based parameter estimation for ESPIRIT
# Ilicak2020 Automated parameter selection for accelerated MRI ...

# TODO: inbounds and size checks

# Espirit: Uecker2013
function espirit_hankel_factorisation(
	calibration::AbstractArray{<: Number, N},
	kernelsize::NTuple{M, Integer}
) where {N, M}
	@assert N == M + 1 # Channels

	#  Get hankel matrix
	hankel = hankel_matrix(calibration, kernelsize)
	num_σ = size(hankel, 2)

	# Multiply Hankel matrix with it's Hermitian conjugate
	# Need the right hand eigen vectors of the Hankel matrix
	# which are the eigen vectors of this matrix
	# Allocate
	hankel_H_hankel = Matrix{ComplexF64}(undef, num_σ, num_σ)
	# Multiply
	BLAS_threads = BLAS.get_num_threads()
	BLAS.set_num_threads(Threads.nthreads())
	mul!(hankel_H_hankel, hankel', hankel)
	BLAS.set_num_threads(BLAS_threads)
	# Make it official
	hankel_H_hankel = Hermitian(hankel_H_hankel)

	# Get factorisation
	factorisation = eigen(hankel_H_hankel) # Here, better use orthogonal iterations
	 # TODO: Better to use n kernels instead of cutting off with a value?
	return factorisation.vectors, factorisation.values
end


"""
	Get kspace kernels

"""
function espirit_select_kernels(
	V::AbstractMatrix{<: Complex},
	σ_squared::AbstractVector{<: Number}, # Must be sorted, ascending
	kernelsize::NTuple{N, Integer},
	σ_cut::Real
) where N

	threshold = σ_cut^2 * σ_squared[end] # the latter is the maximum singular value
	local i
	@inbounds for outer i = length(σ_squared):-1:1
		abs(σ_squared[i]) < threshold && break
	end
	V_parallel = V[:, i:end]
	V_parallel ./= sqrt(prod(kernelsize))
	return V_parallel
end

@generated function espirit_transform_kernels(
	V_parallel::AbstractMatrix{<: Number},
	channels::Integer,
	outshape::NTuple{N, Integer},
	kernelsize::NTuple{N, Integer}
) where N
	
	return quote
		# Join channel and kernel axes for easier access
		neighbours = prod(kernelsize)
		kernels = size(V_parallel, 2)
		channels_and_kernels = channels * kernels
		VH_parallel = begin
			# Restore kspace shape
			tmp = reshape(V_parallel, kernelsize..., channels_and_kernels)
			# Flip back
			# because the found weights are the kernel but flipped (convolution!)
			reverse!(tmp; dims=Tuple(1:$N))
			# Flatten kspace shape and get transpose (better for matrix multiplication below)
			tmp = reshape(tmp, neighbours, channels_and_kernels)
			VH_parallel = (collect ∘ adjoint)(tmp)
		end

		#= TODO: Choose Fourier matrix or fft
			Expense of
				- fft is proportional to sum_{dims} log2(dim) * prod(other dims)
				- Fourier matrix is prod(kernelsizes) * prod(dims)

			Tests showed fft will be faster in practically every 3D case with reasonable output array size.
		=#
		F = fourier_matrix(kernelsize, outshape)

		# Fourier transform filters
		transformed_kernels = Matrix{ComplexF64}(undef, channels_and_kernels, prod(outshape))
		mul!(transformed_kernels, VH_parallel, F)
		# V_∥^H * F = (F^H * V_∥)^H

		# Separate channels and kernels
		return reshape(transformed_kernels, channels, kernels, outshape...) 
	end
end

function espirit_eigen(transformed_kernels::AbstractArray{<: Number, N}, num_v::Integer) where N
	# TODO: power iterations as an alternative
	# Get dimensions
	shape = size(transformed_kernels)
	channels, kernels = shape[1:2]
	spatial_shape = shape[3:N]

	# Allocate space
	v = Array{ComplexF64, N}(undef, channels, spatial_shape..., num_v)
	λ = Array{Float64, N-1}(undef, spatial_shape..., num_v)

	# Iterate spatial positions
	G = [Matrix{ComplexF64}(undef, channels, channels) for _ = 1:Threads.nthreads()]
	@views Threads.@threads for X in CartesianIndices(spatial_shape)
		g = transformed_kernels[:, :, X]
		G[Threads.threadid()] .= g * g'
		factorisation = eigen(Hermitian(G[Threads.threadid()]), channels-num_v+1:channels)
		w = factorisation.vectors
		v[:, X, :] = @. w * exp(-im * angle(w[1:1, :])) # Normalise phase to first channel
		λ[X, :] = factorisation.values
	end
	return v, λ
end


"""
	Transforms only to a single spatial location, to save memory
"""
@generated function espirit_transform_kernels(
	V_parallel::AbstractArray{<: Number, N},
	x::CartesianIndex{M},
	cexp::NTuple{M, Complex},
	partial_ft::NTuple{K, AbstractArray{<: Complex}},
	transformed_kernels::AbstractMatrix{<: Complex}
) where {N, M, K}
	@assert N == M + 2 # Channels and kernels
	@assert M == K + 1 # Since "reduce", first is not required
	return quote
		cexp_x = @ntuple $M (d -> cexp[d] .^ (x[d] - 1))
		for ck in CartesianIndices(size(transformed_kernels)) # channels, kernels
			mini_dft!(cexp_x[1], (@view V_parallel[(@ntuple $M d -> :)..., ck]), partial_ft[1])
			@nexprs $(K-1) d -> begin # First and last iteration need special care
				mini_dft!(cexp_x[d+1], partial_ft[d], partial_ft[d+1])
			end
			transformed_kernels[ck] = mini_dft(cexp_x[$M], partial_ft[$K])
		end
		return
	end
end


"""
	Consumes less memory by splitting the Fourier transform into a voxelwise operation
"""
function espirit_transform_and_eigen(
	V_parallel::AbstractMatrix{<: Number},
	channels::Integer,
	outshape::NTuple{N, Integer},
	kernelsize::NTuple{N, Integer}
) where N
	@assert N > 1
	
	# Join channel and kernel axes for easier access
	neighbours = prod(kernelsize)
	kernels = size(V_parallel, 2)
	length_out = prod(outshape)
	channels_and_kernels = channels * kernels

	V_parallel = conj.(reshape(V_parallel, kernelsize..., channels, kernels))
	#V_parallel = ifftshift(V_parallel, 1:N) Doesn't seem to be correct, don't fully understand why

	# Allocate
	num_threads = Threads.nthreads()
	partial_ft_threaded = ntuple(
		_ -> ntuple(d -> Array{ComplexF64, N-d}(undef, kernelsize[1:N-d]), Val(N-1)),
		num_threads
	)
	# minus two because the first array is not required, and the last is a scalar.
	transformed_kernels_threaded = Array{ComplexF64, 3}(undef, channels, kernels, num_threads) # This is g^T in Uecker2013
	G_threaded = Array{ComplexF64, 3}(undef, channels, channels, num_threads)
	v = Array{ComplexF64, N+1}(undef, channels, outshape...) # Eigen vectors
	λ = Array{Float64, N}(undef, outshape...) # Eigen values
	
	# Precompute exponentials
	cexp = exp.(im * 2π ./ outshape)

	Threads.@threads for x in CartesianIndices(outshape)
		# Get storage for this thread
		@views begin
			tid = Threads.threadid()
			partial_ft = partial_ft_threaded[tid]
			transformed_kernels = transformed_kernels_threaded[:, :, tid]
			G = G_threaded[:, :, tid]
		end
		# Fourier transform kernels
		espirit_transform_kernels(V_parallel, x, cexp, partial_ft, transformed_kernels)
		# Factorise G = g' * g, where g ≡ transformed_kernels^T
		let
			mul!(G, transformed_kernels, transformed_kernels')
			factorisation = eigen(Hermitian(G), channels:channels) # Power iterations are faster here
			λ[x] = factorisation.values[1]
			@views begin
				w = factorisation.vectors[:, 1]
				v[:, x] = w .* exp(-im * angle(w[1])) # Normalise phase to first channel
			end
			# TODO: there is a paper on how to choose the phase so that the image is real (assuming the actual signal distribution is)
		end
	end
	return v, λ
end

function smooth_step(x::Real)
	x ≤ 0 && return 0.0
	x ≥ 1 && return 1.0
	return x^2 * (3 - 2x)
end

"""
	λ_cut should be in (0, 1)
"""
function espirit_mask_eigen(
	v::AbstractArray{<: Number, N},
	λ::AbstractArray{<: Real, M},
	λ_cut::Real,

) where {N,M}
	@assert N == M + 1
	sensitivities = similar(v)
	@views for x in CartesianIndices(λ)
		# Smooth cut-off (sigmoid), see Uecker2013a
		sensitivities[:, x] = @. v[:, x] * smooth_step(λ[x] / λ_cut)
		# Hard cut-off
		#if λ[x] >= λ_cut
		#	sensitivities[:, x] = v[:, x]
		#else
		#	sensitivities[:, x] .= 0
		#end
	end
	return sensitivities
end


"""
Uecker2013
Parameters from the paper: kernelsize=6, σ_cut = 0.001, λ_cut = 0.9 for 24 channels

"""
function estimate_sensitivities(
	method::Val{:ESPIRIT},
	calibration::AbstractArray{<: Number, N},
	outshape::NTuple{M, Integer},
	kernelsize::NTuple{M, Integer},
	σ_cut::Real,
	λ_cut::Real
) where {N, M}
	@assert N > 1
	@assert M == N-1
	
	channels = size(calibration, N)
	V, σ_squared = espirit_hankel_factorisation(calibration, kernelsize)
	V_parallel = espirit_select_kernels(V, σ_squared, kernelsize, σ_cut)
	transformed_kernels = espirit_transform_kernels(V_parallel, channels, outshape, kernelsize)
	v, λ = espirit_eigen(transformed_kernels, 1)
	v = dropdims(v; dims=N+1) # TODO: Check other λ, return multiple sensitivities if required
	λ = dropdims(λ; dims=N)
	sensitivities = espirit_mask_eigen(v, λ, λ_cut)
	sensitivities = permutedims(sensitivities, ((2:N)..., 1)) # Put channels last
	return sensitivities 
end

