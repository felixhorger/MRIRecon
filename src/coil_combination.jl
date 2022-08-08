
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
See Zhang2012
kspace[readout, channels, other spatial dims]
kspace needs to be disentangled along readout direction!
returns compression matrices C[channels, virtual channels, readout] and σ[(virtual) channels, readout], sorted largest to lowest
no cut-off performed!
User needs to choose subset of columns of U
"""
function channel_compression_matrix(kspace::AbstractArray{<: Complex, 3})
	num = size(kspace, 1) # number of independent "slices"
	channels = size(kspace, 2)
	C = Array{ComplexF64, 3}(undef, channels, channels, num)
	σ = Array{Float64, 2}(undef, channels, num)
	@views for i = 1:num
		D = kspace[i, :, :]
		F = eigen(Hermitian(D * D'); sortby=-)
		C[:, :, i] = F.vectors'
		@. σ[:, i] = map((λ -> λ < 0 ? 0 : sqrt(λ)), F.values)
		# Note: if the singular values of D are small, then some of the eigenvalues (small absolute value)
		# of D * D' turn out negative, need to mask those
	end
	return C, σ
end


"""
	σ needs to be sorted largest to lowest
"""
@views function cut_virtual_channels(C::AbstractArray{<: Complex, 3}, σ::AbstractMatrix{<: Real}, ε::Real)
	@assert 0 < ε < 1
	thresholds = ε .* σ[1, :]
	num = size(σ, 2)
	n = 0
	for i = 1:size(σ, 1)
		all(j -> σ[i, j] < thresholds[j], 1:num) && break
		n += 1
	end
	n > size(σ, 1) && error("No compression possible")
	C_cut = Array{ComplexF64, 3}(undef, n, size(C, 2), num)
	C_cut .= C[1:n, :, :]
	return C_cut
end


"""
	C[channels, virtual channels, slices]
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

"""
	kspace[readout spatial (transformed), channels, other spatial dims]
"""
function apply_channel_compression(kspace::AbstractArray{T, 3}, C::AbstractArray{<: Complex, 3}) where T <: Complex
	num = size(kspace, 1)
	#compressed_kspace = similar(kspace, num, size(C, 1), size(kspace, 3))
	# compressed_kspace[virtual channels, other dims, readout]
	channels = size(kspace, 2)
	virtual_channels = size(C, 1)
	other_spatial = size(kspace, 3)
	compressed_kspace = zeros(T, num, virtual_channels, other_spatial)
	for v = 1:virtual_channels, m = 1:other_spatial
		for n = 1:num, c = 1:channels
			@inbounds @fastmath compressed_kspace[n, v, m] += C[v, c, n] * kspace[n, c, m]
		end
	end
	#kspace2_rvs = sum_c C_vcr * kspace_rcs
	#for i = 1:num
	#	@views mul!(compressed_kspace[i, :, :], C[:, :, i], kspace[i, :, :])
	#end
	return compressed_kspace
end

