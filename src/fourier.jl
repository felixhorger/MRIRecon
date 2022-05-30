
function plan_spatial_ft(x::AbstractArray{<: Number, N}, dims::Integer) where N
	# x[spatial dimensions..., other dimensions...]
	# Pay attention that they have to be applied in pairs! Otherwise scaling
	FFT = plan_fft(x, 1:dims)
	FFTH = inv(FFT)
	shape = size(x)
	restore_shape(x) = reshape(x, shape) # Note that x is Vector
	F = LinearMap{ComplexF64}(
		x -> begin
			x = restore_shape(x)
			y = FFT * x
			vec(y)
		end,
		y -> let
			y = restore_shape(y)
			x = FFTH * y
			vec(x)
		end,
		prod(shape)
	)
	return F
end


function plan_spatial_ft(
	kx::AbstractArray{<: Number},
	ky::AbstractArray{<: Number},
	shape::NTuple{3, Integer}
) where N
	# Scaling?

	# Get dims
	num_frequencies = length(kx)
	@assert length(ky) == num_frequencies
	spatial_dims = [shape[1], shape[2]]
	spatial_length = prod(spatial_dims)
	other_dims = shape[3] 

	# Plan NUFFT
	forward_plan = finufft_makeplan(
		2, # type
		spatial_dims,
		-1, # iflag
		other_dims, # ntrans
		1e-8 # eps
	)
	backward_plan = finufft_makeplan(
		1, # type
		spatial_dims,
		1, # iflag
		other_dims, # ntrans
		1e-8 # eps
	)
	for plan in (forward_plan, backward_plan)
		# This should go into the finufft package
		finalizer(plan) do plan
			if finufft_destroy!(plan) != 0
				error("Could not destroy finufft plan")
			end
		end
	end
	normalisation = 1 / spatial_length
	finufft_setpts!(forward_plan, kx, ky)
	finufft_setpts!(backward_plan, kx, ky)

	F = LinearMap{ComplexF64}(
		x -> begin
			x = reshape(x, shape)
			y = finufft_exec(forward_plan, x)
			vec(y)
		end,
		y -> let
			y = reshape(y, num_frequencies, other_dims)
			x = finufft_exec(backward_plan, y) * normalisation
			vec(x)
		end,
		num_frequencies * other_dims, # in
		spatial_length * other_dims # out
	)

	return F
end

