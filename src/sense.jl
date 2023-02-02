
"""
	Size of the operator returned by plan_sensitivities()

	Convenience and reminder, helps keeping scripts clean of excess information.
"""
function sensitivites_size(shape::Tuple{Vararg{Integer}}, num_channels::Integer, num_other::Integer)
	other_length = prod(shape) * num_other
	return (other_length * num_channels, other_length)
end

"""
	sensitivities[spatial dimensions..., channels]
	not side effect free
"""
function plan_sensitivities(
	sensitivities::AbstractArray{T, N},
	num_other::Integer;
	dtype::Type=T,
	Sx::AbstractVector{<: T}=empty(Vector{T}),
	SHy::AbstractVector{<: T}=empty(Vector{T})
) where {T <: Number, N}
	# Get dimensions
	shape = size(sensitivities)
	num_channels = shape[N]
	num_spatial = prod(shape[1:N-1])
	input_dim = num_spatial * num_other
	output_dim = input_dim * num_channels

	# Reshape
	sensitivities = reshape(sensitivities, num_spatial, num_channels)

	# TODO: make a reallocating version of this
	S = LinearOperator{T}(
		(output_dim, input_dim),
		(y, x) -> begin
			xs = reshape(x, num_spatial, num_other)
			ys = reshape(y, num_spatial, num_channels, num_other)
			# TODO turbo
			Threads.@threads for I in CartesianIndices((num_spatial, num_channels, num_other))
				@inbounds ys[I] = sensitivities[I[1], I[2]] * xs[I[1], I[3]]
			end
			y
		end;
		adj = (x, y) -> begin
			# Reset output
			turbo_wipe!(x)
			# Reshape arrays
			xs = reshape(x, num_spatial, num_other)
			ys = reshape(y, num_spatial, num_channels, num_other)
			# Set up threading
			niter = num_spatial * num_other
			nthreads = min(Threads.nthreads(), niter)
			@sync for tid = 1:nthreads
				Threads.@spawn let
					start, stop = ThreadTools.thread_region(tid, niter, nthreads)
					for i = start:stop, c = 1:num_channels
						o, s = divrem(i-1, num_spatial) .+ 1
						# TODO: There is still one cache miss, it would be better to go s, c, o instead of s, o, c
						# but the s-o plane is weirdly cut by threads so need different strategy
						@inbounds xs[s, o] += conj(sensitivities[s, c]) * ys[s, c, o]
					end
				end
			end
			x
		end,
		out=check_allocate(Sx, num_spatial * num_channels * num_other),
		out_adj_inv=check_allocate(SHy, num_spatial * num_other)
	)
	return S
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

