
"""
	plan_sensitivities(sensitivities::Union{AbstractMatrix{<: Number}, AbstractArray{<: Number, 3}}) = S

	sensitivities[spatial dimensions..., channels]
	num_x = all spatial dimensions
"""
function plan_sensitivities(
	sensitivities::AbstractArray{<: Number, N},
	num_x::Integer,
	num_dynamic::Integer
) where N
	# Get dimensions
	shape = size(sensitivities)
	num_channels = shape[N]
	input_dimension = num_x * num_dynamic
	output_dimension = input_dimension * num_channels

	# Reshape
	sensitivities = reshape(sensitivities, num_x, num_channels, 1)
	conj_sensitivities = conj.(sensitivities)

	S = LinearMap{ComplexF64}(
		x::AbstractVector{<: Complex} -> begin
			Sx = sensitivities .* reshape(x, num_x, 1, num_dynamic)
			vec(Sx)
		end,
		y::AbstractVector{<: Complex} -> begin
			y = reshape(y, num_x, num_channels, num_dynamic)
			SHy = sum(conj_sensitivities .* y; dims=2)
			vec(SHy)
		end,
		output_dimension, input_dimension
	)
	return S
end

"""
	plan_PSF([S::LinearMap,] F::LinearMap, M::LinearMap)
	Matrices in the order in which they are applied
	This is more a reminder than really useful
"""
plan_psf(F::LinearMap, M::LinearMap) = F' * M * F
plan_psf(S::LinearMap, F::LinearMap, M::LinearMap) = S' * F' * M * F * S


"""
	shape is spatial
	return psf[spatial dims..., dynamic_out, dynamic_in]
"""
function compute_psf(A::LinearMap, shape::NTuple{N, Integer}, num_dynamic::Integer; fov_scale::Integer=1, fftshifted::Bool=false) where N
	if fftshifted
		centre = ntuple(_ -> 1, Val(N))
	else
		centre = shape .÷ 2
	end
	psf = Array{ComplexF64, N+2}(undef, (fov_scale .* shape)..., num_dynamic, num_dynamic) # out in
	#δ = Vector{ComplexF64}(undef, prod(shape))
	δ = zeros(ComplexF64, prod(shape) * num_dynamic)
	idx = LinearIndices((shape..., num_dynamic))
	colons = ntuple(_ -> :, N+1)
	@views for t = 1:num_dynamic
		i = idx[centre..., t]
		δ[i] = 1 # Set Dirac-delta
		psf[colons..., t] = A * δ
		δ[i] = 0 # Unset
	end
	return psf
end
