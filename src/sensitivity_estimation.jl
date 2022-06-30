


N = 512
channels = 8
sensitivities = zeros(ComplexF64, N, N, channels)
a = ones(ComplexF64, N, N)

# Phantom
a[N÷4:3N÷4, N÷4:3N÷4] .= 1
#a[1:2:end, 1:2:end] .= 1

# Sensitivities
let
	x = zeros(Float64, 2, N, N)
	x[1, :, :] = repeat(1:N, inner=(1, N))
	x[2, :, :] = reshape(repeat(1:N, inner=N), N, N)
	for (c, α) = enumerate(range(0, 2π * (1 - 1/channels); length=channels))
		sine, cosine = sincos(α)
		position = @. 0.25N * [cosine, sine] + 0.5N
		sensitivities[:, :, c] = exp.(-0.5 / (0.25N)^2 .* sum((x .- position).^2; dims=1))
	end
end

calibration = fftshift(fft(a .* sensitivities, (1, 2)), (1,2))




"""
	Uecker2013
	Get kspace kernels

"""
function espirit_kernels(
	calibration::AbstractArray{<: Number, N},
	kernelsize::Integer, # TODO: This could be tuple
	σ_cut::Real
) where N
	@assert N > 1

	hankel = hankel_matrix(calibration, kernelsize)

	# Get linear subspace that is not the null-space (mysterious)
	factorisation = svd!(hankel) # Hankel overwritten
	σ = factorisation.S
	i = 1
	threshold = σ_cut * σ[1] # the latter is the maximum singular value
	for s in σ
		abs(s) < threshold && break
		i += 1
	end
	V_parallel = factorisation.V[:, 1:i]
	V_parallel ./= sqrt(kernelsize^(N-1))
	return V_parallel
end

@generated function espirit_transform_kernels(
	V_parallel::AbstractMatrix{<: Number},
	channels::Integer,
	outshape::NTuple{N, Integer},
	kernelsize::Integer
) where N
	@assert N > 1
	
	return quote

		# Join channel and kernel axes for easier access
		neighbours = kernelsize^$N
		kernels = size(V_parallel, 2)
		channels_and_kernels = channels * kernels
		VT_parallel = begin
			tmp = reshape(V_parallel, neighbours, channels_and_kernels)
			tmp = (collect ∘ transpose)(tmp)
		end

		F = fourier_matrix((@ntuple $N d -> kernelsize), outshape)

		# Fourier transform filters
		length_out = prod(outshape)
		transformed_kernels = Matrix{ComplexF64}(undef, channels_and_kernels, length_out)
		mul!(transformed_kernels, VT_parallel, F) # = F^H * conj.(kernels)

		# Separate channels and kernels
		return reshape(transformed_kernels, channels, kernels, length_out) 
	end
end

function espirit_eigen(transformed_kernels::AbstractArray{<: Number, 3})
	# Get dimensions
	channels, kernels, length_out = size(transformed_kernels)

	# Allocate space
	eigenvecs = Array{ComplexF64, 3}(undef, channels, channels, length_out)
	eigenvals = Array{Float64, 2}(undef, channels, length_out)

	# Iterate spatial positions
	g = Matrix{ComplexF64}(undef, channels, kernels)
	@inbounds @views for x = 1:length_out
		g .= transformed_kernels[:, :, x]
		factorisation = svd!(g) # Equivalent to the eigen decomposition of G = g * g'
		eigenvecs[:, :, x] = factorisation.U
		eigenvals[:, x] = factorisation.S
	end
	return eigenvecs, eigenvals
end

function espirit_mask_eigenvecs(
	eigenvecs::AbstractArray{<: Number, 3},
	eigenvals::AbstractArray{<: Real, 2},
	λ_cut::Real
)
	@views begin
		# Synchronise relative phase via the first channel
		sensitivities = @. eigenvecs[:, 1, :] * exp(-im * angle(eigenvecs[1:1, 1, :]))
		# Mask where λ ≠ 1
		mask = @. (eigenvals[1, :])^2 > λ_cut
		@inbounds for i in eachindex(mask)
			mask[i] || (sensitivities[:, i] .= 0)
		end
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
	kernelsize::Integer,
	σ_cut::Real,
	λ_cut::Real
) where {N, M}
	@assert N > 1
	@assert M == N-1
	
	channels = size(calibration, N)
	V_parallel = espirit_kernels(calibration, kernelsize, σ_cut)
	transformed_kernels = espirit_transform_kernels(V_parallel, channels, outshape, kernelsize)
	eigenvecs, eigenvals = espirit_eigen(transformed_kernels)
	sensitivities = espirit_mask_eigenvecs(eigenvecs, eigenvals, λ_cut)
	sensitivities = reshape(sensitivities, channels, outshape...)
	return sensitivities
end


"""
	First axis of must be channels
"""
@inline function espirit_project(data::AbstractArray{<: Number, N}, sensitivities::AbstractArray{<: Number, N}) where N
	conj.(sensitivities) .* sum(data .* sensitivities; dims=1)
end

#plt.figure()
#plt.imshow(abs.(x))
#plt.show()

