
function plan_spatial_ft(
	x::AbstractArray{<: Number, N} where N,
	dims::Union{Integer, NTuple{M, <: Integer} where M, AbstractVector}
)
	# Best practice: x[spatial dimensions..., other dimensions...]
	# If the user decides to mix spatial and other dimensions, good luck!
	# Pay attention that they have to be applied in pairs! Otherwise scaling
	FFT = plan_fft(x, dims)
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
	# TODO: Version where only the powers are stored and matrix is subtype of AbstractMatrix, getindex computes element on the fly
	@assert N > 0
	return quote
		shift = inshape .÷ 2
		# Precompute exponentials
		@nexprs $N d -> max_power_d = (outshape[d] .- 1) .* shift[d]
		@nexprs $N d -> ξ_d = @. exp(-im * 2π / outshape[d])^(-max_power_d:max_power_d)
		fourier_matrix = Array{ComplexF64, $(2N)}(undef, inshape..., outshape...)
		# Fill matrix
		for X in CartesianIndices(outshape) # Spatial positions
			x = Tuple(X) .- 1
			for K in CartesianIndices(inshape) # Iterate over neighbours
				k = Tuple(K) .- shift .- 1
				fourier_matrix[K, X] = prod(@ntuple $N d -> ξ_d[x[d] * k[d] + max_power_d + 1])
			end
		end
		return reshape(fourier_matrix, prod(inshape), prod(outshape))
	end
end



"""
	Evaluate a single Fourier coefficient
	Not as accurate as the fft since the Fourier sum is explicitly computed

	cexp = exp(im * 2π / size(f, 1))^x

"""
# TODO: Inbounds
@generated function mini_dft!(
	cexp::Complex,
	f::AbstractArray{<: Number, N},
	F::AbstractArray{<: Complex, M}
) where {N,M}
	@assert N == M + 1
	return quote
		@views for x in CartesianIndices(size(f)[2:$N])
			F[x] = evalpoly(cexp, f[:, x])
		end
		return F
	end
end
mini_dft(cexp::Complex, f::AbstractVector{<: Number}) = evalpoly(cexp, f)


"""
	Mathematically equivalent to fftshift, but in kspace
"""
function shift_half_fov!(kspace::AbstractArray{<: Number}, dims)
	dims = Set(dims)
	return shift_half_fov!(kspace, dims)
end
function shift_half_fov!(kspace::AbstractArray{<: Number, N}, dims::Set{<: Integer}) where N
	@assert all(d -> d <= N, dims)
	for K in CartesianIndices(kspace)
		s = 0
		for d in dims
			s += K[d]
		end
		if isodd(s)
			kspace[K] *= -1
		end
	end
	return kspace
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


import FFTW: fftshift, ifftshift
for (func, op) in ( (:fftshift, :(.+)), (:ifftshift, :(.-)) )
	@eval begin
		function $func(
			indices::AbstractVector{<: CartesianIndex{N}},
			shape::NTuple{N, Integer}
		) where N
			indices = copy(indices)
			shift = shape .÷ 2
			for i in eachindex(indices)
				indices[i] = CartesianIndex(
					(mod1(
						$(Expr(:call, op, :(indices[i][j]), :(shift[j]))), # index ± shift
						shape[j]
					) for j = 1:N)...
				)
			end
			return indices
		end
	end
end


"""
	Circular convolution
	This should be available in DSP.jl
"""
function circular_conv(u::AbstractArray{<: Number}, v::AbstractArray{<: Number})
	# TODO: Pad to same size
	F = plan_fft(u)
	FH = inv(F)
	FH * ((F * u) .* (F * v))
end

