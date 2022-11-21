
"""
	Not side effect free
	Best practice: x[spatial dimensions..., other dimensions...]
	If the user decides to mix spatial and other dimensions, good luck!
	Pay attention that they have to be applied in pairs! Otherwise scaling
"""
function plan_fourier_transform!(
	FHy::Array{T, N},
	dims::Union{Integer, NTuple{M, Integer} where M, AbstractVector{<: Integer}};
	type::Type{T}=ComplexF64,
	kwargs...
) where {N, T}

	# TODO: write Fx also in Fhy and rename it to sth more generic, this is for the case when x shall not be mutated

	# Arrays for in-place operations
	Fx = similar(FHy)
	vec_Fx = vec(Fx)
	vec_FHy = vec(FHy)

	# Plan FFTs
	FFT = plan_fft(Fx, dims; kwargs...)
	FFTH = plan_ifft(FHy, dims; kwargs...)

	# Reshaping
	shape = size(FHy)
	restore_shape(z) = reshape(z, shape) # Note that z is vector

	# Define linear operator
	F = UnitaryOperator(
		prod(shape),
		x -> begin
			x = restore_shape(x)
			mul!(Fx, FFT, x)
			vec_Fx
		end,
		y -> begin
			y = restore_shape(y)
			mul!(FHy, FFTH, y)
			vec_FHy
		end
	)
	return F
end
function plan_fourier_transform!(
	shape::NTuple{N, Integer},
	dims::Union{Integer, NTuple{M, Integer} where M, AbstractVector{<: Integer}};
	type::Type{T}=ComplexF64,
	kwargs...
) where {N, T}

	# Plan FFTs
	if haskey(kwargs, :flags) && kwargs[:flags] in (FFTW.MEASURE, FFTW.PATIENT)
		z = Array{T, N}(undef, shape)
	else
		z = FFTW.FakeArray{T, N}(shape, cumprod((1, shape[1:N-1]...)))
	end
	FFT = plan_fft!(z, dims; kwargs...)
	FFTH = plan_ifft!(z, dims; kwargs...)

	# Reshaping and views
	restore_shape(z) = reshape(z, shape) # Note that z is a vector here

	# Define linear operator
	F = UnitaryOperator(
		prod(shape),
		x -> begin
			FFT * restore_shape(x) # In place
			x
		end,
		y -> begin
			FFTH * restore_shape(y) # In place
			y
		end
	)
	return F
end


"""
	Not side effect free
"""
function plan_fourier_transform(k1, k2, shape::NTuple{3, Integer}; eps::Real=1e-8, dtype::Type{T}=Float64, kwargs...) where T
	FHy = Array{Complex{T}, 3}(undef, shape)
	return plan_fourier_transform!(FHy, k1, k2, shape; dtype, kwargs...)
end
function plan_fourier_transform!(
	FHy::Array{Complex{T}, 3},
	k1::Vector{<: Number},
	k2::Vector{<: Number},
	shape::NTuple{3, Integer};
	eps::Real=1e-8,
	dtype::Type{T}=Float64,
	upsampling=2,
	kwargs...
) where T

	# Get dims
	num_frequencies = length(k1)
	@assert length(k2) == num_frequencies
	spatial_dims = [shape[1], shape[2]]
	spatial_length = prod(spatial_dims)
	other_dims = shape[3] 
	@assert size(FHy) == shape

	# Plan NUFFT
	forward_plan = finufft_makeplan(
		2, # type
		spatial_dims,
		-1, # iflag
		other_dims, # ntrans
		eps; # eps
		dtype,
		upsampfac=upsampling,
		kwargs...
	)
	backward_plan = finufft_makeplan(
		1, # type
		spatial_dims,
		1, # iflag
		other_dims, # ntrans
		eps; # eps
		dtype,
		upsampfac=upsampling,
		kwargs...
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
	finufft_setpts!(forward_plan, k1, k2)
	finufft_setpts!(backward_plan, k1, k2)

	# Allocate output
	Fx = Array{Complex{T}, 2}(undef, num_frequencies, other_dims)
	vec_Fx = vec(Fx)
	vec_FHy = vec(FHy)

	# Linear operators
	F = LinearOperator(
		(num_frequencies * other_dims, spatial_length * other_dims),
		x -> begin
			x = reshape(x, shape)
			finufft_exec!(forward_plan, x, Fx)
			vec_Fx
		end,
		y -> begin
			y = reshape(y, num_frequencies, other_dims)
			finufft_exec!(backward_plan, y, FHy)
			# TODO: LoopVectorization
			Threads.@threads for i in eachindex(FHy)
				@inbounds FHy[i] *= normalisation
			end
			vec_FHy
		end
	)
	return F
end
function plan_toeplitz_embedding(
	k1::Vector{<: Number},
	k2::Vector{<: Number},
	shape::NTuple{3, Integer};
	dtype::Type{T}=Float64,
	weights::AbstractVector{<: Number}=ones(Complex{dtype}, length(k1)),
	eps::Real=1e-8,
	upsampling::Integer=2,
	finufft_kwargs::Dict=Dict(),
	fftw_flags=FFTW.MEASURE
) where T
	@assert length(k1) == length(k2) == length(weights)
	upsampled_shape = upsampling .* shape[1:2]
	num_x = prod(shape[1:2])

	# Compute convolution function (essentially point-spread-function)
	q = nufft2d1(
		k1, k2,
		(weights ./ num_x),
		1,
		eps,
		upsampled_shape...;
		dtype,
		modeord=1, # Required to get the "fftshifted" version
		upsampfac=upsampling,
		finufft_kwargs...
	)
	# Plan convolution operator
	C, y_padded = plan_circular_convolution((upsampled_shape..., shape[3]), q, 1:2; flags=fftw_flags)

	# Allocate memory
	x_padded = Array{Complex{T}, 3}(undef, upsampled_shape..., shape[3])
	vec_x_padded = vec(x_padded)
	y = Array{Complex{T}, 3}(undef, shape)
	vec_y = vec(y)
	centre = MRIRecon.centre_indices.(upsampled_shape, shape[1:2])
	centre_of_y_padded = @view y_padded[centre..., :]

	# Toeplitz Embedding
	FHF = HermitianOperator(
		num_x * shape[3],
		x -> begin
			@inbounds x_padded[centre..., :] = x
			vec_y_padded = C * vec_x_padded
			# Basically: y_padded = reshape(vec_y_padded, upsampled_shape..., shape[3])
			@inbounds vec_y[:] = centre_of_y_padded
			vec_y
		end
	)
	return FHF
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
function shift_half_fov!(kspace::AbstractArray{<: Number, N}, dims) where N
	@assert allunique(dims)
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


function shift_fov!(kspace::AbstractArray{<: Number}, shift::AbstractVector{<: Real})
	shift_fov!(kspace, Tuple(shift)) # Calling it that way might pay off because then the callee can be compiled
	# for that dimension
end
function shift_fov!(
	kspace::AbstractArray{<: Number, N},
	shift::NTuple{D, Real}
) where {N, D}
	@assert D ≤ N
	# Precompute phase modulation
	exponentials = exp.(-im .* shift)
	shape = size(kspace)[1:D]
	phase_modulation = Array{ComplexF64, D}(undef, shape)
	offset = .- shape .÷ 2
	scale = 2π ./ shape
	for k in CartesianIndices(shape)
		exponent = scale .* (Tuple(k) .+ offset)
		phase_modulation[k] = prod(exponentials .^ exponent)
	end
	kspace .*= phase_modulation
	return kspace
end
function shift_fov!(
	kspace::AbstractArray{<: Number},
	k::AbstractMatrix{<: Real}, # First axis x,y,... second axis samples
	shift::AbstractVector{<: Real}
)
	@assert size(k, 1) == length(shift) # Spatial dimension
	@assert size(kspace, 1) == size(k, 2) # Number of samples
	# Precompute phase modulation
	phase_modulation = exp.(-im .* (k' * shift))
	kspace .*= phase_modulation
	return kspace
end


"""
	Sinc interpolation downsampling, in the first M axes
"""

function downsample(a::AbstractArray{T, N}, downsampling::NTuple{M, Integer}) where {N, M, T <: Number}
	shape = size(a)[1:M]
	downsampled_shape = divrem.(shape, downsampling)
	@assert all(p -> p[2] == 0, downsampled_shape) "downsampling must yield no remainder when dividing the size of a (except last dimension)"
	downsampled_shape = ntuple(d -> downsampled_shape[d][1], M)
	# Fourier transform
	b = fftshift(fft(a, 1:M), 1:M)
	# Extract centre indices, cutting off high frequency components
	centre_indices = MRIRecon.centre_indices.(shape, downsampled_shape)
	b = @view b[centre_indices..., ntuple(_ -> :, N-M)...]
	# Transform back
	interpolated = ifft(ifftshift(b, 1:M), 1:M) ./ prod(downsampling)
	return interpolated
end

function upsample!(
	a::AbstractArray{T, N},
	shape::NTuple{M, Integer},
	outshape::NTuple{M, Integer},
	residual_shape::NTuple{K, Integer}
) where {N, M, K, T <: Complex}
	b = zeros(T, outshape..., residual_shape...)
	centre_indices = MRIRecon.centre_indices.(outshape, shape)
	b[centre_indices..., ntuple(_ -> :, N-M)...] = fftshift(fft(a, 1:M), 1:M)
	# Transform back
	interpolated = ifft(ifftshift(b, 1:M), 1:M)
	return interpolated
end
function upsample!(
	a::AbstractArray{T, N},
	shape::NTuple{M, Integer},
	outshape::NTuple{M, Integer},
	residual_shape::NTuple{K, Integer}
) where {N, M, K, T <: Real}
	b = zeros(complex(T), outshape[1]÷2+1, outshape[2:M]..., residual_shape...)
	# Zero pad the Fourier transform of a
	b[1:shape[1]÷2+1, ntuple(d -> 1:shape[d+1], M-1)..., ntuple(_ -> :, N-M)...] = rfft(a, 1:M)
	# Transform back
	interpolated = irfft(b, outshape[1], 1:M)
	return interpolated
end
"""
	Sinc interpolation upsampling, in the first M axes

	Note, this "shifts" the origin of the volume by half an index in the upsampled space!
	The occupied volume is thus slightly different.
	Draw it on a piece of paper and you'll see it makes sense!
"""
function upsample(a::AbstractArray{<: Number, N}, outshape::NTuple{M, Integer}) where {N, M}
	shape = size(a)[1:M]
	residual_shape = size(a)[M+1:N]
	upsampling = outshape ./ shape
	interpolated = upsample!(a, shape, outshape, residual_shape) .* prod(upsampling)
	return interpolated
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

		function $func(
			indices::AbstractVector{<: CartesianIndex{N}},
			shape::NTuple{N, Integer},
			dims
		) where {N,M}
			indices = copy(indices)
			shift = shape .÷ 2
			tmp = Vector{Int64}(undef, N)
			for i in eachindex(indices)
				tmp .= Tuple(indices[i])
				for d in dims
					tmp[d] = mod1(
						$(Expr(:call, op, :(tmp[d]), :(shift[d]))), # index ± shift
						shape[d]
					)
				end
				indices[i] = CartesianIndex(tmp...)
			end
			return indices
		end
	end
end


"""
	Not side effect free
"""
function plan_circular_convolution(shape::NTuple{N, Integer}, c::AbstractArray{<: Number, M}, dims::Union{Integer, NTuple{D, Integer}, AbstractVector}; kwargs...) where {N, M, D}
	# Array for in-place operations
	Fx = Array{ComplexF64, N}(undef, shape)
	vec_xc = vec(Fx)
	# Plan Fourier
	F = plan_fft(Fx, dims; kwargs...)
	FH = plan_bfft!(Fx, dims; kwargs...) # In-place
	# Precompute kernel, cannot use F in general due to different shapes.
	Fc = fft(c, dims) ./ prod(shape[1:N-1]) # fft normalisation included here
	# Define convolution operator
	C = HermitianOperator(
		prod(shape),
		x -> begin
			mul!(Fx, F, reshape(x, shape))
			# Compute Fx .*= Fc
			for J in CartesianIndices(shape[M+1:N])
				Threads.@threads for I in CartesianIndices(Fc)
					@inbounds Fx[I, J] *= Fc[I]
				end
			end
			xc = FH * Fx # in-place, so just a change of names
			vec_xc
		end
	)
	return C, Fx
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


"""
kspace[fully sampled and disentangled dims or other (e.g. channels), partially sampled spatial dims...]
"""
function partial_fourier(kspace::AbstractArray{<: C, N}, num_calibration::Integer, unsampled::AbstractVector{<: CartesianIndex{M}}) where {N, M, C <: Complex}
	@assert N == M + 1
	# Get dimensions
	shape = size(kspace)[2:N]
	calibration_indices = centre_indices.(shape, num_calibration)
	# Plan fft
	FH = plan_bfft!(kspace, 2:N) # don't care for normalisation
	F = inv(FH)
	# Get k-space centre
	calibration = zeros(C, size(kspace))
	calibration[:, calibration_indices...] .= kspace[:, calibration_indices...]
	# Get phase map
	phase_factor = ifftshift(calibration, 2:N) # misnomer phase_factor
	FH * phase_factor # in-place
	phase_factor = phasefactor.(phase_factor)
	# Invert k-space
	kspace = ifftshift(kspace, 2:N) # misnomer imagespace
	imagespace = FH * kspace # in-place
	# Remove phase
	imagespace ./= phase_factor
	# Transform back
	symmetric_kspace = F * imagespace # in-place
	# Mirror part of k-space
	# Get the negative k:
	#=
	K	->  I - centre
	-K	->  centre - I
	-I	->  -K + centre = centre - I + centre = 2centre - I
	=#
	offset = CartesianIndex(2 .* (shape .÷ 2 .+ 1))
	shift = shape .÷ 2
	fftshift(I) = CartesianIndex(mod1.(Tuple(I) .- shift, shape))
	for I in unsampled
		J = offset - I
		(I, J) = fftshift.((I, J))
		@views symmetric_kspace[:, I] = conj.(symmetric_kspace[:, J])
	end
	# Add phase back in
	imagespace = FH * symmetric_kspace # in-place
	imagespace .*= phase_factor ./ prod(shape)
	return imagespace
end

