
"""
	sensitivities[spatial dimensions..., channels]
	not side effect free
"""
function plan_sensitivities(
	sensitivities::AbstractArray{<: Number, N},
	num_dynamic::Integer
) where N
	# Get dimensions
	shape = size(sensitivities)
	num_channels = shape[N]
	num_spatial = prod(shape[1:N-1])
	input_dim = num_spatial * num_dynamic
	output_dim = input_dim * num_channels

	# Reshape
	sensitivities = reshape(sensitivities, num_spatial, num_channels)

	# Allocate
	Sx = Array{ComplexF64, 3}(undef, num_spatial, num_channels, num_dynamic)
	SHy = Array{ComplexF64, 2}(undef, num_spatial, num_dynamic)
	vec_Sx = vec(Sx)
	vec_SHy = vec(SHy)

	# TODO: make a reallocating version of this
	S = LinearOperator(
		(output_dim, input_dim),
		x -> begin
			x = reshape(x, num_spatial, num_dynamic)
			for t = 1:num_dynamic, c = 1:num_channels
				Threads.@threads for i = 1:num_spatial
					@inbounds Sx[i, c, t] = sensitivities[i, c] * x[i, t]
				end
			end
			vec_Sx
		end,
		y -> begin
			y = reshape(y, num_spatial, num_channels, num_dynamic)
			SHy .= 0
			for n = 1:num_dynamic, c = 1:num_channels
				Threads.@threads for i = 1:num_spatial
					@inbounds SHy[i, n] += conj(sensitivities[i, c]) * y[i, c, n]
				end
			end
			vec_SHy
		end
	)
	return S, reshape(Sx, shape..., num_dynamic)
end

"""
	plan_PSF(F::AbstractLinearOperator, M::AbstractLinearOperator [, S::AbstractLinearOperator])
"""
plan_psf(M::AbstractLinearOperator, F::AbstractLinearOperator) = F' * M * F
plan_psf(M::AbstractLinearOperator, F::AbstractLinearOperator, S::AbstractLinearOperator) = S' * F' * M * F * S


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

