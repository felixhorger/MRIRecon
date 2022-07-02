
function root_sum_of_squares(signals::AbstractArray{<: Number}, dim::Val{N}) where N
	result = dropdims(sum(abs2, signals; dims=N); dims=N)
	@. result = sqrt(result)
end

function roemer(
	signals::AbstractArray{<: Number},
	sensitivity::AbstractArray{<: Number},
	dim::Val{N}=Val(1)
) where N
	dropdims(
		sum(conj.(sensitivity) .* signals; dims=N) ./ sum(abs2, sensitivity; dims=N);
		dims=N
	)
end



"""
	plan_sensitivities(sensitivities::Union{AbstractMatrix{<: Number}, AbstractArray{<: Number, 3}}) = S

sensitivities[spatial dimensions..., channels]
shape = (spatial dimensions, other dimension)

"""
function plan_sensitivities(
	sensitivities::AbstractArray{<: Number, N},
	shape::NTuple{M, Integer}
) where {N,M}
	@assert M == N + 1

	# Get dimensions
	# TODO: check spatial dims
	other_dims = shape[M]
	shape = shape[1:N]
	shape_s = size(sensitivities)
	@assert shape == shape_s
	channels = shape_s[N]
	spatial_dimensions = prod(shape[1:N-1])
	input_dimension = spatial_dimensions * other_dims
	output_dimension = input_dimension * channels

	# Reshape
	sensitivities = reshape(sensitivities, spatial_dimensions, channels, 1)
	conj_sensitivities = conj.(sensitivities)

	S = LinearMap{ComplexF64}(
		x::AbstractVector{<: Complex} -> begin
			Sx = sensitivities .* reshape(x, spatial_dimensions, 1, other_dims)
			vec(Sx)
		end,
		y::AbstractVector{<: Complex} -> begin
			y = reshape(y, spatial_dimensions, channels, other_dims)
			SHy = sum(conj_sensitivities .* y; dims=2)
			vec(SHy)
		end,
		output_dimension, input_dimension
	)
	return S
end

