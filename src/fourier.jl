
# TODO: offer unnormalised Fourier transform, normalisation can be included in masking
"""
	Not side effect free
	Best practice: x[spatial dimensions..., other dimensions...]
	If the user decides to mix spatial and other dimensions, good luck!
	Pay attention that they have to be applied in pairs! Otherwise scaling
	TODO: problem if F' F is applied, because F writes into y, and F' too.
	However, F' F = I, so why bother?
"""
function plan_fourier_transform(
	y::AbstractArray{<: T, N}, # TODO: make this argument analogous to the other operators
	dims::Union{Integer, NTuple{M, Integer} where M, AbstractVector{<: Integer}};
	kwargs...
) where {T <: Complex, N}

	# Plan FFTs
	FFT = plan_fft(y, dims; kwargs...)
	FFTH = plan_ifft(y, dims; kwargs...)

	# Reshaping
	shape = size(y)

	# Define linear operator
	F = UnitaryOperator{T}(
		prod(shape),
		(y, x) -> begin
			mul!(reshape(y, shape), FFT, reshape(x, shape))
			y
		end;
		adj = (x, y) -> begin
			mul!(reshape(x, shape), FFTH, reshape(y, shape))
			x
		end,
		out=vec(y)
	)
	return F
end

"""
In-place
"""
function plan_fourier_transform!(
	shape::NTuple{N, Integer},
	dims::Union{Integer, NTuple{M, Integer} where M, AbstractVector{<: Integer}};
	type::Type{T}=ComplexF64,
	kwargs...
) where {N, T <: Complex}

	# Plan FFTs
	# TODO: only use actual array, also other functions!
	if haskey(kwargs, :flags) && kwargs[:flags] in (FFTW.MEASURE, FFTW.PATIENT)
		z = Array{T, N}(undef, shape)
	else
		z = FakeArray{T, N}(shape, cumprod((1, shape[1:N-1]...)))
	end
	FFT = plan_fft!(z, dims; kwargs...)
	FFTH = plan_ifft!(z, dims; kwargs...)

	# Define linear operator
	F = UnitaryOperator{T}(
		prod(shape),
		(y, x) -> begin
			@assert length(y) == 0
			FFT * reshape(x, shape) # In place
			x
		end;
		adj = (x, y) -> begin
			@assert length(x) == 0
			FFTH * reshape(y, shape) # In place
			y
		end
	)
	return F
end

# TODO: unified concept for putting the shape first for all plan_...() functions

function fourier_transform_size(num_frequencies::Integer, shape::NTuple{D, Integer}) where D
	spatial_length = prod(shape[1:D-1])
	other_dims = shape[D]
	num_in = spatial_length * other_dims
	num_out = num_frequencies * other_dims
	return (num_out, num_in)
end


function plan_fourier_transform(
	k::Matrix{<: Number}, # ndims x num_frequencies
	shape::NTuple{D, Integer};
	dtype::Type{T}=ComplexF64,
	Fx::AbstractVector{<: T}=empty(Vector{dtype}),
	FHy::AbstractVector{<: T}=empty(Vector{dtype}),
	eps::Real=1e-12,
	upsampling=2,
	nthreads::Integer=Threads.nthreads(),
	kwargs...
) where {D, T <: Complex}

	@assert size(k, 1) ∈ (1, 2, 3)
	num_frequencies = size(k, 2)
	num_out, num_in = fourier_transform_size(num_frequencies, shape)
	spatial_dims = [shape[1:D-1]...]
	other_dims = shape[D]

	# Allocate output
	Fx = check_allocate(Fx, num_out)
	FHy = check_allocate(FHy, num_in)

	# Plan NUFFT
	forward_plan = finufft_makeplan(
		2, # type
		spatial_dims,
		-1, # iflag
		other_dims, # ntrans
		eps; # eps
		dtype=real(T),
		upsampfac=upsampling,
		nthreads,
		kwargs...
	)
	backward_plan = finufft_makeplan(
		1, # type
		spatial_dims,
		1, # iflag
		other_dims, # ntrans
		eps; # eps
		dtype=real(T),
		upsampfac=upsampling,
		nthreads,
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
	normalisation = 1 / prod(spatial_dims)
	k_split = collect.(eachrow(k))
	finufft_setpts!(forward_plan, k_split...)
	finufft_setpts!(backward_plan, k_split...)

	# Linear operators
	F = LinearOperator{T}(
		(num_out, num_in),
		(y, x) -> begin
			finufft_exec!(forward_plan, reshape(x, shape), reshape(y, num_frequencies, other_dims))
			y
		end;
		adj = (x, y) -> begin
			finufft_exec!(backward_plan, reshape(y, num_frequencies, other_dims), reshape(x, shape))
			xd = decomplexify(x)
			turbo_multiply!(xd, normalisation; num_threads=Val(nthreads))
			x
		end,
		out=Fx,
		out_adj_inv=FHy
	)
	return F
end


# Convolution
function apply_convolution_kernel!(Fx::AbstractMatrix{<: Complex}, Fcd::AbstractMatrix{<: Real})
	@assert size(Fcd, 2) == size(Fx, 1)
	Fxd = decomplexify(Fx)
	@tturbo for j in axes(Fxd, 3), i in axes(Fxd, 2)
		v_re = Fxd[1, i, j]
		v_im = Fxd[2, i, j]
		w_re = Fcd[1, i]
		w_im = Fcd[2, i]
		Fxd[1, i, j] = v_re * w_re - v_im * w_im
		Fxd[2, i, j] = v_re * w_im + v_im * w_re
	end
	return Fx
	#for J in CartesianIndices(shape[M+1:N])
	#	Threads.@threads for I in CartesianIndices(Fc)
	#		@inbounds @fastmath Fx[I, J] *= Fc[I]
	#	end
	#end
end

function prepare_periodic_convolution(y::AbstractArray{T, N}, c::AbstractArray{T, M}; kwargs...) where {T <: Number, N, M}
	shape = size(y)
	F = plan_fft!(y, 1:M; kwargs...)
	FH_unnormalised = plan_bfft!(y, 1:M; kwargs...) # In-place
	# Precompute kernel, cannot use F in general due to different shapes.
	Fc = fft(c, 1:M) ./ prod(shape[1:M]) # fft normalisation included here
	Fcd = decomplexify(vec(Fc)) # This is a Matrix
	# Define convolution operator
	y_matrix_shape = (prod(shape[1:M]), prod(shape[M+1:N]))
	return F, FH_unnormalised, Fcd, y_matrix_shape
end

"""
	Not side effect free
	Only complex data due to type stability
"""
# TODO: Real version?
function plan_periodic_convolution(
	y::AbstractArray{<: T, N}, # Output
	c::AbstractArray{<: Number, M};
	kwargs...
) where {T <: Complex, N, M}
	# Array for in-place operations
	F, FH_unnormalised, Fcd, y_matrix_shape = prepare_periodic_convolution(y, c; kwargs...)
	shape = size(y)
	C = HermitianOperator{T}(
		prod(shape),
		(y, x) -> begin
			Fx = reshape(y, shape)
			# Fourier transform
			mul!(Fx, F, reshape(x, shape))
			# Multiply with transformed kernel
			apply_convolution_kernel!(reshape(Fx, y_matrix_shape), Fcd)
			# Transform back
			FH_unnormalised * Fx # in-place
			y
		end;
		out=vec(y)
	)
	return C, Fx
end

function plan_periodic_convolution!(shape::NTuple{N, Integer}, c::AbstractArray{T, M}; kwargs...) where {T <: Number, N, M}
	# Plan Fourier
	if haskey(kwargs, :flags) && kwargs[:flags] in (FFTW.MEASURE, FFTW.PATIENT)
		y = Array{T, N}(undef, shape)
	else
		y = FakeArray{T, N}(shape, cumprod((1, shape[1:N-1]...)))
	end
	F, FH_unnormalised, Fcd, y_matrix_shape = prepare_periodic_convolution(y, shape, c; kwargs...)
	C = HermitianOperator{complex(T)}(
		prod(shape),
		(y, x) -> begin
			@assert length(y) == 0
			# Fourier transform
			Fx = F * reshape(x, shape) # in-place
			# Multiply with transformed kernel
			apply_convolution_kernel!(reshape(Fx, y_matrix_shape), Fcd)
			# Transform back
			FH_unnormalised * Fx # in-place
			x
		end
	)
	return C
end

"""
	Circular convolution
	This should be available in DSP.jl
"""
function periodic_conv(u::AbstractArray{<: Number}, v::AbstractArray{<: Number})
	# TODO: Pad to same size
	F = plan_fft(u)
	FH = inv(F)
	FH * ((F * u) .* (F * v))
end




function apply_toeplitz!(
	y::AbstractArray{T, N},
	x::AbstractArray{T, N},
	padded::AbstractArray{T, N},
	C::HermitianOperator{T}
) where {T <: Complex, N}
	# Dimensions and checks
	shape = size(x)[1:N-1] # First axis is length 2, decomplexified
	double_shape = size(padded)[1:N-1]
	num_other = size(x, N)
	@assert num_other == size(padded, N)
	# Reshape and decomplexify arrays
	(x_d, y_d) = decomplexify.((x, y))
	padded_d = decomplexify(padded)
	# Zero fill padded array
	@tturbo for i in 1:length(padded_d)
		padded_d[i] = 0
	end
	# Pad x into padded
	zero_offset = Tuple(0 for _ = 1:N+1)
	offset = (0, centre_offset.(shape)..., 0)
	shape_d = (2, shape..., num_other)
	turbo_block_copyto!(padded_d, x_d, shape_d, offset, zero_offset)
	# Convolve
	C * vec(padded) # in-place
	# Crop into y
	turbo_block_copyto!(y_d, padded_d, shape_d, zero_offset, offset)
	return y
end

# TODO: Upsampling can only be two here actually, the user must ensure everything is in the FOV!
toeplitz_padded_size(shape::NTuple{3, Integer}; upsampling::Integer=2) = ((upsampling .* shape[1:2])..., shape[3])
function prepare_toeplitz(
	padded::AbstractArray{<: T, 3},
	k1::Vector{<: Number},
	k2::Vector{<: Number},
	weights::AbstractVector{<: Number},
	shape::NTuple{3, Integer}, # (spatial 1, spatial 2, other)
	dtype::Type{T},
	eps::Real,
	upsampling::Integer,
	finufft_kwargs::Dict,
	fftw_flags;
	inverse::Bool=false
) where T <: Complex
	@assert length(k1) == length(k2) == length(weights)
	upsampled_shape = toeplitz_padded_size(shape, upsampling)
	# Compute convolution function (essentially point-spread-function)
	q = nufft2d1(
		k1, k2,
		(weights ./ prod(shape[1:2])),
		1,
		eps,
		upsampled_shape[1:2]...;
		dtype=real(T),
		modeord=1, # Required to get the "fftshifted" version
		upsampfac=upsampling,
		finufft_kwargs...
	)
	if inverse
		@. q = 1 / q
	end
	# Plan convolution operator
	C = plan_periodic_convolution!(upsampled_shape, dropdims(q; dims=3); flags=fftw_flags)
	# Preallocate output (if necessary)
	padded = check_allocate(padded, upsampled_shape)
	return C, padded
end
function plan_toeplitz(
	y::AbstractArray{<: T, 3},
	k1::Vector{<: Number}, # FINUFFT doesn't like anything but actual vectors
	k2::Vector{<: Number},
	weights::AbstractVector{<: Number};
	dtype::Type{T}=ComplexF64,
	padded::AbstractArray{<: T, 3}=empty(Array{T, 3}),
	eps::Real=1e-12,
	upsampling::Integer=2,
	finufft_kwargs::Dict=Dict(),
	fftw_flags=FFTW.MEASURE,
	inverse::Bool=false
) where T <: Complex
	shape = size(y)
	# Prepare convolution and padded array
	C, padded = prepare_toeplitz(padded, k1, k2, weights, shape, dtype, eps, upsampling, finufft_kwargs, fftw_flags; inverse)
	# Toeplitz Embedding operator
	FHMF = HermitianOperator{T}(
		prod(shape),
		(y, x) -> begin
			apply_toeplitz!(reshape(y, shape), reshape(x, shape), padded, C)
			y
		end;
		out=vec(y)
	)
	return FHMF
end

function plan_toeplitz!(
	k1::Vector{<: Number},
	k2::Vector{<: Number},
	weights::AbstractVector{<: Number},
	shape::NTuple{3, Integer}; # (spatial 1, spatial 2, other)
	dtype::Type{T}=ComplexF64,
	padded::AbstractArray{<: T, 3}=empty(Array{T, 3}),
	eps::Real=1e-12,
	upsampling::Integer=2,
	finufft_kwargs::Dict=Dict(),
	fftw_flags=FFTW.MEASURE,
	inverse::Bool=false
) where T <: Complex
	# Prepare padded array and convolution
	C, padded = prepare_toeplitz(padded, k1, k2, weights, shape, dtype, eps, upsampling, finufft_kwargs, fftw_flags; inverse)
	# Toeplitz Embedding
	FHMF = HermitianOperator{T}(
		prod(shape),
		(y, x) -> begin
			@assert length(y) == 0
			xs = reshape(x, shape)
			apply_toeplitz!(xs, xs, padded, F, FH_unnormalised)
			y
		end
	)
	return FHMF
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
# TODO: check sizes and inbounds
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


shift_fov!(kspace::AbstractArray{<: Number}, shift::AbstractVector{<: Real}) = shift_fov!(kspace, Tuple(shift))
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

# TODO: why the hell did I put '!' here?
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
		upsample(a::AbstractArray{<: Number, N}, outshape::NTuple{M, Integer}) where {N, M}

	Sinc interpolation upsampling, in the first M axes

	Note, this "shifts" the origin of the volume by half an index in the upsampled space!
	The occupied volume is thus slightly different.
	Draw it on a piece of paper and you'll see it makes sense!

	I think it actually shifts a full index
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
					ntuple(
						j -> mod1(
							$(Expr(:call, op, :(indices[i][j]), :(shift[j]))), # index ± shift
							shape[j]
						),
						N
					)
				)
			end
			return indices
		end

		function $func(
			indices::AbstractVector{<: CartesianIndex{N}},
			shape::NTuple{N, Integer},
			dims
		) where N
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


# TODO: really need vector of unsampled indices? Not entirely sure which cases make sense here
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

