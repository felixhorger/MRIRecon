# Uecker2019 ENLIVE
# Uecker2017 ESPIRIT+Virtual Conjugate Coils
# Iyer2020 Sure based parameter estimation for ESPIRIT
# Ilicak2020 Automated parameter selection for accelerated MRI ...

# TODO: inbounds and size checks

# Espirit: Uecker2013
function espirit_hankel_factorisation(
	calibration::AbstractArray{<: Number, N}, # [spatial dims..., channels]
	kernelsize::NTuple{M, Integer},
	σ_cut::Real
) where {N, M}
	error("hankel updated")
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
	mul!(hankel_H_hankel, hankel', hankel)
	# Make it official
	hankel_H_hankel = Hermitian(hankel_H_hankel)

	# Get factorisation
	#factorisation = eigen(hankel_H_hankel, num_σ-num_kernels+1:num_σ) # Here, better use orthogonal iterations?
	# Here, better use orthogonal iterations?
	σ_max = (first ∘ eigvals)(hankel_H_hankel, num_σ:num_σ)
	factorisation = eigen(hankel_H_hankel, σ_cut^2 * σ_max, Inf)
	V_parallel = factorisation.vectors ./ sqrt(prod(kernelsize))
	return V_parallel, factorisation.values
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
	@assert σ_squared[1] < threshold "Threshold too small ($(σ_squared[1]) < $threshold)"
	i = length(σ_squared)
	@inbounds while i > 1 && σ_squared[i] ≥ threshold
		i -= 1
	end
	i += 1
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

function espirit_eigen(transformed_kernels::AbstractArray{<: Number, N}, num_λ::Integer) where N
	# TODO: power iterations as an alternative
	# Get dimensions
	shape = size(transformed_kernels)
	channels, kernels = shape[1:2]
	spatial_shape = shape[3:N]

	# Allocate space
	v = Array{ComplexF64, N}(undef, channels, num_λ, spatial_shape...)
	λ = Array{Float64, N-1}(undef, num_λ, spatial_shape...)

	# Iterate spatial positions
	which_eigenvecs = channels-num_λ+1:channels
	Gs = [Matrix{ComplexF64}(undef, channels, channels) for _ = 1:Threads.nthreads()]
	Threads.@threads for X in CartesianIndices(spatial_shape)
		G = Gs[Threads.threadid()]
		g = @view transformed_kernels[:, :, X]
		G .= g * g'
		factorisation = eigen(Hermitian(G), which_eigenvecs)
		# TODO: This isn't alright. I guess it indicates degeneracy
		#if length(factorisation.values) == 0
		#	factorisation = eigen(G)
		#	v[:, X, :] = factorisation.vectors[:, 1:num_v]
		#	λ[X, :] = factorisation.values[1:num_v]
		#else
			v[:, :, X] = factorisation.vectors
			λ[:, X] = factorisation.values
		#end
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
	λ_cut must be in (0, 1)
"""
function espirit_mask_eigen(
	v::AbstractArray{<: Number, N},
	λ::AbstractArray{<: Real, M},
	λ_cut::Real,
	λ_tol::Real
) where {N,M}
	@assert N == M + 1
	sensitivities = similar(v)
	inv_λ_tol = 1 / λ_tol
	@views for x in CartesianIndices(λ)
		# Smooth cut-off (sigmoid), see Uecker2013a
		sensitivities[:, x] = @. v[:, x] * smooth_step((λ[x] - λ_cut + λ_tol) * inv_λ_tol)
		# Hard cut-off
		#if λ[x] >= λ_cut
		#	sensitivities[:, x] = v[:, x]
		#else
		#	sensitivities[:, x] .= 0
		#end
	end
	return sensitivities
end

function espirit_eigenmaps(
	calibration::AbstractArray{<: Number, N}, # spatial dims, channels
	outshape::NTuple{M, Integer},
	kernelsize::NTuple{M, Integer},
	σ_cut::Real,
	num_λ::Integer
) where {N, M}
	@assert N > 1
	@assert M == N-1
	
	channels = size(calibration, N)
	V, σ_squared = espirit_hankel_factorisation(calibration, kernelsize, σ_cut)
	V_parallel = espirit_select_kernels(V, σ_squared, kernelsize, σ_cut)
	transformed_kernels = espirit_transform_kernels(V_parallel, channels, outshape, kernelsize)
	v, λ = espirit_eigen(transformed_kernels, num_λ)
	return v, λ
end


function phasefactor(a::T) where T <: Number
	absolute = abs(a)
	return absolute == zero(T) ? one(T) : a / absolute
end

function complex_sign(a::Complex)
	r, i = reim(a)
	if r ≠ 0
		return sign(r)
	else
		return sign(i)
	end
end

@generated function align_sign!(a::AbstractArray{T, N}, maxiter) where {T <: Number, N}
	return quote
		#@assert all(d -> d ≤ N, dims)
		updated = true
		while updated && maxiter > 0
			updated = false
			@nloops $N i a begin
				v = phasefactor(@nref $N a i)
				@nloops $N j d -> mod1.(i_d-1:2:i_d+1, size(a, d)) begin
					w = phasefactor(@nref $N a j)
					if abs2(v - w) > abs2(v + w)
						(@nref $N a j) *= -1
						updated = true
					end
				end
			end
			maxiter -= 1
		end
		return
	end
end



"""
calibration[spatial dims..., channels]
"""
function espirit_add_conjugated_channels(calibration::AbstractArray{T, N}) where {T <: Number, N}
	M = N - 1
	shape = size(calibration)[1:M]
	num_channels = size(calibration, N)
	# Get kspaces and flipped kspaces (virtual channels)
	vcc_calibration = Array{T, N}(undef, shape..., 2num_channels)
	l = length(calibration)
	vcc_calibration[1:l] = calibration
	spatial_indices = CartesianIndices(shape)
	tmp = @. conj(calibration)
	reverse!(tmp; dims=ntuple(d -> d, M))
	@views circshift!(vcc_calibration[spatial_indices, num_channels+1:2num_channels], tmp, ntuple(_ -> -1, M))
	return vcc_calibration
end

"""
	reference must be interpolated from a low-resolution reference scan 
	calibration is non-vcc calibration!
	v[spatial dims... channels]
"""
function espirit_align_phase!(
	v::AbstractArray{<: Number, N},
	calibration::AbstractArray{<: Number, M}
) where {N, M}
	@assert N > 1
	L = M-1

	shape = size(v)[2:N] # Includes the "λ" dimension (last)
	calibration_shape = size(calibration)[1:L]
	num_channels = size(calibration, M)
	@assert size(v, 1) == 2num_channels

	# Undo arbitrary phase shift inherent to ESPIRIT
	for x in CartesianIndices(shape)
		s = 0.0
		for c = 1:num_channels
			s += v[c, x] * v[c+num_channels, x]
		end
		# In the paper s = exp(2iφ + n⋅2π) n ∈ Z
		v[1:num_channels, x] .*= 2 * (sqrt ∘ conj ∘ phasefactor)(s)
		# Factor two because eigenvectors in v are normalised to twice the number of channels
	end
	# Truncate virtual channels
	v = @view v[1:num_channels, CartesianIndices(shape)]
	return v
	# At this point, only the sign is unclear

	# Get low-res phase factor
	#lowres_phasefactor = estimate_sensitivities(Val(:Direct), calibration, shape)
	#@. lowres_phasefactor = (conj ∘ phasefactor)(lowres_phasefactor) # exp(-iϕ)
	#lowres_phasefactor = permutedims(lowres_phasefactor, (N, 1:M...)) # TODO this is hmmm

	## Compute the unknown signs
	#signs = @. (v * lowres_phasefactor) |> complex_sign

	# Revert signs
	#v .*= signs
	for c = 1:num_channels
		align_sign!(v[c, CartesianIndices(shape)], 100)
		@show c
	end

	return v
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

	v, λ = espirit_eigenmaps(calibration, outshape, kernelsize, σ_cut)
	sensitivities = espirit_mask_eigen(v, λ, λ_cut)
	sensitivities = permutedims(sensitivities, ((2:N)..., 1)) # Put channels last
	return sensitivities 
end

function estimate_sensitivities(
	method::Val{:VCCESPIRIT},
	calibration::AbstractArray{T, N},
	outshape::NTuple{M, Integer},
	kernelsize::NTuple{M, Integer},
	σ_cut::Real,
	λ_cut::Real
) where {T <: Number, N, M}
	@assert N > 1
	@assert M == N-1

	shape = size(calibration)[1:M]
	num_channels = size(calibration, N)

	vcc_calibration = espirit_add_conjugated_channels(calibration)
	v, λ = espirit_eigenmaps(vcc_calibration, outshape, kernelsize, σ_cut)

	sensitivities = espirit_mask_eigen(v, λ, λ_cut)
	sensitivities = permutedims(sensitivities, ((2:N)..., 1)) # Put channels last
	return sensitivities 
end





function compute_window(
	shape::NTuple{N, Integer},
	window::Function
) where N
	windows_dict = Dict{Int64, Vector{Float64}}()
	window_lengths = unique(shape)
	for n in window_lengths
		windows_dict[n] = window(n)
	end
	windows = Vector{Vector{Float64}}(undef, N)
	for d = 1:N
		windows[d] = windows_dict[shape[d]]
	end
	return windows
end

function apply_window!(
	a::AbstractArray{<: Number, N},
	windows::AbstractVector{<: Vector{<: Number}}
) where N
	# Apply window function
	for k in CartesianIndices(size(a))
		for d = 1:N
			a[k] *= windows[d][k[d]]
		end
	end
	return a
end

function direct_unnormalised_sensitivities(
	calibration::AbstractArray{T, N},
	outshape::NTuple{M, Integer}
) where {N, M, T <: Number}
	@assert M == N-1
	shape = size(calibration)[1:M]
	num_channels = size(calibration, N)

	calibration = copy(calibration)

	# Windowing
	windows = compute_window(shape, n -> DSP.Windows.kaiser(n, 4)) 
	spatial_indices = CartesianIndices(shape)
	for c = 1:num_channels
		@views apply_window!(
			calibration[spatial_indices, c],
			windows
		)
	end

	# Pad to outshape and inverse Fourier transform
	padded = zeros(T, outshape..., num_channels)
	centre_indices = MRIRecon.centre_indices.(outshape, shape)
	padded[centre_indices..., :] = calibration
	sensitivities = ifft(ifftshift(padded, 1:M), 1:M)
	rsos = root_sum_of_squares(sensitivities; dim=N)
	return sensitivities, rsos
end

# TODO: Rename to rsos so that it's more clear
function direct_normalise_sensitivities!(
	sensitivities::AbstractArray{<: Number, N},
	rsos::AbstractArray{<: Real, M},
	cut::Real,
	tol::Real
) where {N, M}
	@assert M == N-1 # Channels are not in rsos or outshape
	@assert 0 ≤ cut ≤ 1
	@assert 0 < tol

	# Normalise with root sum of squares
	maxi = maximum(rsos)
	cut *= maxi
	tol *= maxi
	inv_tol = 1 / tol
	for x in CartesianIndices(rsos)
		normalisation = rsos[x]
		attenuation = smooth_step((normalisation - cut + tol) * inv_tol)
		if normalisation == 0
			factor = 0.0
		else
			factor = attenuation / normalisation
		end
		sensitivities[x, :] .*= factor
	end
	return sensitivities
end

"""
	1) Apply window to calibration in kspace (remove Gibbs ringing)
	2) Inverse Fourier transform
	3) Divide by root sum of squares
"""
function estimate_sensitivities(
	method::Val{:Direct},
	calibration::AbstractArray{T, N},
	outshape::NTuple{M, Integer},
	cut::Real,
	tol::Real
) where {N, M, T <: Number}

	sensitivities, rsos = direct_unnormalised_sensitivities(calibration, outshape)
	sensitivities = direct_normalise_sensitivities!(sensitivities, rsos, cut, tol)
	return sensitivities
end

