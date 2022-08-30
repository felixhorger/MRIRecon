
"""
	sensitivities[spatial dimensions..., channels]
"""
function plan_sensitivities(
	sensitivities::AbstractArray{<: Number, N},
	num_dynamic::Integer
) where N
	# Get dimensions
	shape = size(sensitivities)
	num_channels = shape[N]
	num_x = prod(shape[1:N-1])
	input_dimension = num_x * num_dynamic
	output_dimension = input_dimension * num_channels

	# Reshape
	sensitivities = reshape(sensitivities, num_x, num_channels, 1)

	S = LinearMap{ComplexF64}( # Note: This is not really julian! Type is checked in LinearMap :(
		x::AbstractVector{ComplexF64} -> begin
			Sx = sensitivities .* reshape(x, num_x, 1, num_dynamic)
			vec(Sx)
		end,
		y::AbstractVector{ComplexF64} -> begin
			y = reshape(y, num_x, num_channels, num_dynamic)
			SHy = zeros(ComplexF64, num_x, num_dynamic)
			for n = 1:num_dynamic, c = 1:num_channels
				Threads.@threads for x = 1:num_x
					@inbounds SHy[x, n] += conj(sensitivities[x, c, 1]) * y[x, c, n]
				end
			end
			vec(SHy)
		end,
		output_dimension, input_dimension
	)
	return S
end

"""
	plan_PSF(F::LinearMap, M::LinearMap [, S::LinearMap])
"""
plan_psf(M::LinearMap, F::LinearMap) = F' * M * F
plan_psf(M::LinearMap, F::LinearMap, S::LinearMap) = S' * F' * M * F * S


"""
	A must be callable, taking a vector to compute the matrix vector product
	shape is spatial
	return psf[spatial dims..., dynamic_out, dynamic_in]
"""
function compute_psf(
	A,
	shape::NTuple{N, Integer},
	num_dynamic::Integer;
	fov_scale::Integer=1,
	fftshifted::Bool=false
) where N

	if fftshifted
		centre = ntuple(_ -> 1, Val(N))
	else
		centre = shape .÷ 2
	end
	psf = Array{ComplexF64, N+2}(undef, (fov_scale .* shape)..., num_dynamic, num_dynamic) # out in
	δ = zeros(ComplexF64, prod(shape) * num_dynamic)
	idx = LinearIndices((shape..., num_dynamic))
	colons = ntuple(_ -> :, N+1)
	@views for t = 1:num_dynamic
		i = idx[centre..., t]
		δ[i] = 1 # Set Dirac-delta
		psf[colons..., t] = A(δ)
		δ[i] = 0 # Unset
	end
	return psf
end

