
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
	kx::Vector{<: Number},
	ky::Vector{<: Number},
	shape::NTuple{3, Integer},
	eps::Real,
	dtype::Type
)

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
		eps; # eps
		dtype
	)
	backward_plan = finufft_makeplan(
		1, # type
		spatial_dims,
		1, # iflag
		other_dims, # ntrans
		eps; # eps
		dtype
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


@generated function fourier_matrix(inshape::NTuple{N, Integer}, outshape::NTuple{N, Integer}) where N
	@assert N > 0
	return quote
		shift = inshape .÷ 2
		# Precompute exponentials
		@nexprs $N d -> max_power_d = (outshape[d] .- 1) .* shift[d]
		@nexprs $N d -> ξ_d = @. exp(-im * 2π / outshape[d])^(-max_power_d:max_power_d)
		fourier_matrix = Array{ComplexF64, $(2N)}(undef, inshape..., outshape...)
		# Fill matrix
		@inbounds @nloops $N x (d -> 0:outshape[d]-1) begin # Spatial positions
			x = (@ntuple $N d -> x_d)
			@nloops $N k (d -> -shift[d]:shift[d]-1) begin # Iterate over neighbours
				k = (@ntuple $N d -> k_d)
				fourier_matrix[(k .+ shift .+ 1)..., (x .+ 1)...] = prod(@ntuple $N d -> ξ_d[x_d * k_d + max_power_d + 1])
			end
		end
		return reshape(fourier_matrix, prod(inshape), prod(outshape))
	end
end


@generated function half_fov_shift!(kspace::AbstractArray{<: Number, N}, first_n_axes::Val{M}) where {N,M}
	return quote
		@inbounds @nloops $N k kspace begin # nloops gives k_1, k_2, ... k_N
			s = 0
			@nexprs $M (d -> s += k_d) # This only considers k_1, k_2, ... k_M
			if isodd(s)
				(@nref $N kspace k) *= -1
			end
		end
		return kspace
	end
end


function shift_fov!(
	kspace::AbstractMatrix{<: Number},
	k::AbstractMatrix{<: Real}, # First axis x,y,... second axis samples
	shift::AbstractVector{<: Real}
)
	@assert size(k, 1) == length(shift) # Spatial dimension
	@assert size(kspace, 1) == size(k, 2) # Number of samples
	# Precompute phase modulation
	phase_modulation = exp.(-im .* (k' * shift))
	# Iterate
	@inbounds for i in axes(kspace, 2)
		kspace[:, i] .*= phase_modulation
	end
end

