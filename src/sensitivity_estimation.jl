


N = 512
channels = 8
sensitivities = zeros(ComplexF64, N, N, channels)
a = ones(ComplexF64, N, N)

# Phantom
a[N÷4:3N÷4, N÷4:3N÷4] .= 1
#a[1:2:end, 1:2:end] .= 1

# Sensitivities
let
	x = zeros(Float64, 2, N, N)
	x[1, :, :] = repeat(1:N, inner=(1, N))
	x[2, :, :] = reshape(repeat(1:N, inner=N), N, N)
	for (c, α) = enumerate(range(0, 2π * (1 - 1/channels); length=channels))
		sine, cosine = sincos(α)
		position = @. 0.25N * [cosine, sine] + 0.5N
		sensitivities[:, :, c] = exp.(-0.5 / (0.25N)^2 .* sum((x .- position).^2; dims=1))
	end
end

calibration = fftshift(fft(a .* sensitivities, (1, 2)), (1,2))



"""
Uecker2013
Parameters from the paper: kernelsize=6, σ_cut = 0.001, λ_cut = 0.9 for 24 channels
"""
@generated function estimate_sensitivities(
	method::Val{:ESPIRIT},
	calibration::AbstractArray{<: Number, N},
	outshape::NTuple{M, Integer},
	kernelsize::Integer,
	σ_cut::Real,
	λ_cut::Real
) where {N, M}
	@assert N > 1
	@assert M == N - 1

	return quote
		@assert all(outshape .> 1)
		@assert iseven(kernelsize)

		# Get dimensions
		calibration_shape = size(calibration)
		kspace_shape = calibration_shape[1:$M]
		channels = calibration_shape[N]
		length_out = prod(outshape)
		neighbours = kernelsize^$M
		matrix_columns = neighbours * channels
		neighbour_shift = kernelsize ÷ 2
		neighbour_range = -neighbour_shift:neighbour_shift-1

		# Construct calibration matrix
		A = let
			A = Array{ComplexF64, N}(undef, kspace_shape..., matrix_columns) 
			@inbounds @nloops $M k calibration begin # Iterate over spatial positions
				# Reset counter for neighbours and channels
				i = 1 
				# Get tuple of spatial indices
				k = @ntuple $M d -> k_d
				for $(Symbol("l_$N")) = 1:channels # Sneaky
					@nloops $M l (d -> neighbour_range) (d -> l_d += k_d) begin # Iterate over neighbours
						# Shift to current position
						l = @ntuple $M d -> l_d + k_d
						# Check if outside of array
						if (@nany $M d -> l_d < 1) || (@nany $M d -> l_d > kspace_shape[d])
							A[k..., i] = 0 # Outside
						else
							A[k..., i] = @nref $N calibration l
						end
						i += 1
					end
				end
			end
			# Flatten spatial dimensions to make it a matrix
			reshape(A, prod(kspace_shape), matrix_columns)
		end

		# Get linear subspace that is not the null-space (mysterious)
		local VH_parallel, kernels
		let
			factorisation = svd!(A) # Don't need A anymore
			σ = factorisation.S
			i = 1
			threshold = σ_cut * maximum(σ)
			for s in σ
				abs(s) < threshold && break
				i += 1
			end
			V_parallel = factorisation.V[:, 1:i-1]
			kernels = size(V_parallel, 2)
			VH_parallel = collect(reshape(V_parallel, neighbours, channels * kernels)')
		end

		transformed_filters = let
			# Construct Fourier matrix
			fourier_matrix = let
				# Precompute exponentials
				@nexprs $M d -> max_power_d = (outshape[d] .- 1) .* neighbour_shift
				@nexprs $M d -> ξ_d = @. exp(im * 2π / outshape[d])^(-max_power_d:max_power_d)
				fourier_matrix = Array{ComplexF64, $(2M)}(undef, (@ntuple $M d -> kernelsize)..., outshape...)
				# Fill matrix
				@inbounds @nloops $M x (d -> 0:outshape[d]-1) begin # Spatial positions
					x = (@ntuple $M d -> x_d)
					@nloops $M k (d -> neighbour_range) begin # Iterate over neighbours
						k = (@ntuple $M d -> k_d)
						fourier_matrix[(k .+ (neighbour_shift + 1))..., (x .+ 1)...] = prod(@ntuple $M d -> ξ_d[x_d * k_d + max_power_d + 1])
					end
				end
				reshape(fourier_matrix, neighbours, length_out)
			end
			# Get Fourier transformed filters
			transformed_filters = Matrix{ComplexF64}(undef, channels * kernels, length_out)
			mul!(transformed_filters, VH_parallel, fourier_matrix)
			transformed_filters = reshape(transformed_filters, channels, kernels, outshape...)
			transformed_filters ./= sqrt(neighbours)
		end

		# Solve eigenvalue problem
		local eigenvecs, eigenvals
		let
			G = Matrix{ComplexF64}(undef, channels, channels)
			eigenvecs = Array{ComplexF64, 3}(undef, channels, channels, length_out)
			eigenvals = Array{ComplexF64, 2}(undef, channels, length_out)
			transformed_filters = reshape(transformed_filters, channels, kernels, length_out)
			@inbounds for x = 1:length_out
				g = @view transformed_filters[:, :, x]
				G .= conj.(g * g')
				factorisation = eigen(G)
				eigenvecs[:, :, x] = factorisation.vectors
				eigenvals[:, x] = factorisation.values
			end
		end

		# Compute sensitivities by masking with the eigenvalues
		@views begin
			mask = abs2.(eigenvals[channels:channels, :]) .> λ_cut
			sensitivities = mask .* eigenvecs[:, channels, :]
			sensitivities = reshape(sensitivities, channels, outshape...)
		end
		return sensitivities
	end
end


#plt.figure()
#plt.imshow(abs.(x))
#plt.show()

