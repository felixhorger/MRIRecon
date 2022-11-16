
function root_sum_of_squares(imagespace::AbstractArray{<: Number}; dim::Integer)
	result = dropdims(sum(abs2, imagespace; dims=dim); dims=dim)
	@. result = sqrt(result)
end

function roemer(
	imagespace::AbstractArray{<: Number},
	sensitivity::AbstractArray{<: Number};
	dim::Integer
)
	dropdims(
		sum(conj.(sensitivity) .* imagespace; dims=dim) ./ sum(abs2, sensitivity; dims=dim);
		dims=dim
	)
end


"""
See Zhang2012
kspace[channels, spatial dims]
returns compression matrices C[virtual channels, channels] and σ[(virtual) channels], sorted largest to lowest
no cut-off performed!
User needs to choose subset of columns of C
"""
function channel_compression_matrix(D::AbstractMatrix{<: Complex})
	channels = size(D, 1)
	C = Matrix{ComplexF64}(undef, channels, channels)
	σ = Vector{Float64}(undef, channels)
	channel_compression_matrix!(C, σ, D)
	return C, σ
end
function channel_compression_matrix!(
	C::AbstractMatrix{<: Complex},
	σ::AbstractVector{<: Real},
	D::AbstractMatrix{<: Complex} # data matrix D[channels, other dims]
)
	mul!(C, D, D') # Use memory provided by user
	F = eigen(Hermitian(C); sortby=-)
	C .= F.vectors'
	map!((λ -> λ < 0 ? 0 : sqrt(λ)), σ, F.values)
	# Note: if the singular values of D are small, then some of the eigenvalues (small absolute value)
	# of D * D' turn out negative, need to mask those
	return C, σ
end


"""
	σ needs to be sorted largest to lowest
	ε is the relative threshold w.r.t. the maximum eigenvalue
	q is which quantile the number of it must be above

	One function for one fully sampled and disentangled dimension (3-dim C)
	Another for full 3D compression (2-dim C)
"""
function cut_virtual_channels(C::AbstractArray{<: Complex, 3}, σ::AbstractMatrix{<: Real}, ε::Real, q::Real)
	@assert 0 < ε < 1
	@views thresholds = ε .* σ[1, :]
	num = size(σ, 2)
	required_num_smaller = Int(ceil(q * num))
	i = 1
	while i ≤ size(σ, 1)
		num_smaller = 0
		for j = 1:num
			if σ[i, j] < thresholds[j]
				num_smaller += 1
				if num_smaller ≥ required_num_smaller
					@goto finish
				end
			end
		end
		i += 1
	end
	@label finish
	i == 1+size(σ, 1) && error("No compression possible")
	C_cut = C[1:i-1, :, :]
	return C_cut
end
function cut_virtual_channels(C::AbstractMatrix{<: Complex}, σ::AbstractVector{<: Real}, ε::Real)
	@assert 0 < ε < 1
	threshold = ε .* first(σ)
	i = 1
	while i ≤ length(σ)
		if σ[i] < threshold
			break
		end
		i += 1
	end
	i == 1+length(σ) && error("No compression possible")
	C_cut = C[1:i-1, :]
	return C_cut
end


"""
	C[virtual channels, channels, slices]
	returns C_aligned
"""
#, R
#where R is the "rotation" matrices which can be used for other scans of a similar setup
#in the paper R is denoted P, however this isn't a projection according to my understanding, so misnomer!
function align_channels(C::AbstractArray{<: Complex, 3})
	# Get dimensions
	virtual_channels, channels, num = size(C)
	# Allocate memory
	#TODO, needed? R = Array{ComplexF64, 3}(undef, virtual_channels, virtual_channels, num)
	R_tmp = Matrix{ComplexF64}(undef, virtual_channels, virtual_channels) 
	C_aligned = Array{ComplexF64, 3}(undef, virtual_channels, channels, num)
	# Set initial values
	# TODO: R[:, :, 1] = I(virtual_channels)
	C_aligned[:, :, 1] = @view C[:, :, 1]
	# Iterate
	@views for i = 2:num
		# Compute the product of compression matrices from consecutive slices
		W = C_aligned[:, :, i-1] * C[:, :, i]'
		F = svd(W)
		# Note: in the paper the adjoint of W is used,
		# but if the SVD is W^H = U Σ V^H then W = V Σ U^H (Σ is diagonal and real)
		# Compute the rotation
		mul!(R_tmp, F.U, F.Vt) # This is then V U^H like in the paper
		# Put "rotated" matrix into the "aligned" array
		mul!(C_aligned[:, :, i], R_tmp, C[:, :, i])
		# Accumulate rotation TODO: not sure if needed
		#mul!(R[:, :, i], R_tmp, R[:, :, i-1])
	end
	return C_aligned #, R
end

"""
	Applied the accumulated rotation matrices R to U in order to align them
	TODO: useful?
"""
function align_channels(C::AbstractArray{<: Complex, 3}, R::AbstractArray{<: Complex, 3})
	virtual_channels, channels, num = size(C)
	@assert size(R) == (virtual_channels, virtual_channels, num)
	C_aligned = Array{ComplexF64, 3}(undef, virtual_channels, channels, num)
	for i = 1:num
		@views mul!(C_aligned[:, :, i], R[:, :, i], C[:, :, i])
	end
	return C_aligned
end

convenient_compression_matrix(C::AbstractArray{<: Number, 3}, dim::Val{1}) = permutedims(C, (2, 1, 3))
convenient_compression_matrix(C::AbstractArray{<: Number, 3}, dim::Val{2}) = permutedims(C, (3, 2, 1)) 
convenient_compression_matrix(C::AbstractArray{<: Number, 3}, dim::Val{3}) = permutedims(C, (2, 3, 1)) 

"""
	kspace[channels, spatial dims, slices]
	C[channels, virtual_channels, slices]
	dim is where the channels are
"""
function apply_channel_compression(
	kspace::AbstractArray{T, 3},
	C::AbstractArray{<: Complex, 3},
	dim::Val{1}
) where T <: Complex
	num_channels, num_spatial, num_slices = size(kspace)
	num_virtual_channels = size(C, 2)
	@assert size(C, 1) == num_channels
	@assert size(C, 3) == num_slices
	compressed_kspace = zeros(T, num_virtual_channels, num_spatial, num_slices)
	Threads.@threads for n = 1:num_slices
		for v = 1:num_virtual_channels, m = 1:num_spatial, c = 1:num_channels
			@inbounds compressed_kspace[v, m, n] += C[c, v, n] * kspace[c, m, n]
		end
	end
	return compressed_kspace
end

"""
	kspace[slices, channels, spatial dims]
	C[slices, channels, virtual_channels]
	dim is where the channels are
"""
function apply_channel_compression(
	kspace::AbstractArray{T, 3},
	C::AbstractArray{<: Complex, 3},
	dim::Val{2}
) where T <: Complex
	num_slices, num_channels, num_spatial = size(kspace)
	num_virtual_channels = size(C, 3)
	@assert size(C, 2) == num_channels
	@assert size(C, 1) == num_slices
	compressed_kspace = zeros(T, num_slices, num_virtual_channels, num_spatial)
	Threads.@threads for m = 1:num_spatial
		for v = 1:num_virtual_channels, c = 1:num_channels, n = 1:num_slices
			@inbounds compressed_kspace[n, v, m] += C[n, c, v] * kspace[n, c, m]
		end
	end
	return compressed_kspace
end
"""
	kspace[spatial dims, slices, channels]
	C[channels, virtual_channels, slices]
	dim is where the channels are
"""
function apply_channel_compression(
	kspace::AbstractArray{T, 3},
	C::AbstractArray{<: Complex, 3},
	dim::Val{3}
) where T <: Complex
	num_spatial, num_slices, num_channels = size(kspace)
	num_virtual_channels = size(C, 2)
	@assert size(C, 1) == num_channels
	@assert size(C, 3) == num_slices
	compressed_kspace = zeros(T, num_spatial, num_slices, num_virtual_channels)
	Threads.@threads for v = 1:num_virtual_channels
		for c = 1:num_channels, n = 1:num_slices, m = 1:num_spatial
			@inbounds compressed_kspace[m, n, v] += C[c, v, n] * kspace[m, n, c]
		end
	end
	return compressed_kspace
end

function apply_channel_compression(
	kspace::AbstractMatrix{T},
	C::AbstractMatrix{<: Complex}
) where T <: Complex
	num_channels, num_spatial = size(kspace)
	num_virtual_channels = size(C, 1)
	@assert size(C, 2) == num_channels
	compressed_kspace = zeros(T, num_virtual_channels, num_spatial)
	mul!(compressed_kspace, C, kspace)
	return compressed_kspace
end


function apply_channel_compression(
	kspace::AbstractArray{T, N},
	C::AbstractArray{<: Complex, 3},
	dims::Val{NTuple{N, Integer}}
) where {T <: Complex, N}
	# dims indicates which dimension is what (spatial, channel slice). What about C?
	num_spatial, num_slices, num_channels = size(kspace)
	num_virtual_channels = size(C, 2)
	@assert size(C, 1) == num_channels
	@assert size(C, 3) == num_slices
	compressed_kspace = zeros(T, num_spatial, num_slices, num_virtual_channels)
	for v = 1:num_virtual_channels, c = 1:num_channels, n = 1:num_slices, m = 1:num_spatial
		compressed_kspace[m, n, v] += C[c, v, n] * kspace[m, n, c]
	end
	return compressed_kspace
end
