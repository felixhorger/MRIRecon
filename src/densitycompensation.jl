import FunctionZeros: besselj_zero
using Bessels
import SpecialFunctions
using QuadGK
# TODO: use import only, not using. Or, can do import ...: fct()

function kernel_integral(kernel::Function, N::Integer, limit::Real)
	quadgk(k -> kernel(k) * k, 0, limit, rtol=1e-8)[1] * SpecialFunctions.gamma(N/2) / (2π^(N/2))
	# Note: (2π)^(N/2) / gamma(N/2) is unit sphere surface area
	# TODO: why is there 1/2π where it should be ⋅2π?
end


# Points outside shape will be clamped to nearest (Boston-distance) inside compartment
function divide_into_chunks(
	samples::AbstractMatrix{<: Real},
	chunks::NTuple{N, Integer}
) where N
	@assert size(samples, 1) == N

	@show chunksize = 2π ./ chunks

	grid = Array{Vector{Int}, N}(undef, chunks)
	n_uniform = size(samples, 2) ÷ prod(chunks)
	for i in eachindex(grid)
		v = Vector{Int}(undef, 0)
		sizehint!(v, n_uniform)
		grid[i] = v
	end

	for s in axes(samples, 2)
		@views i = clamp.(
			floor.(Int, (Tuple(samples[:, s]) .+ π) ./ chunksize) .+ 1,
			1, chunks
		)
		push!(grid[i...], s)
	end

	return grid
end


function generate_neighbour_pairs(chunks::NTuple{N, Integer}) where N
	@assert all(chunks .> 2)
	num_pairs = (3^N - 1) * prod(chunks) #prod((c-1) * 2 + 1 for c in chunks) # Empirical formula
	neighbour_pairs = Vector{NTuple{2, CartesianIndex{N}}}(undef, 0)
	sizehint!(neighbour_pairs, num_pairs)
	@show neighbours = [
		one(CartesianIndex{N}),
		CartesianIndex(2, ntuple(i -> 1, N-1)...),
		(
			CartesianIndex((Tuple(I) .- 1)..., 2, ntuple(i -> 1, N-d)...)
			for d = 2:N
			for I in CartesianIndices(ntuple(i -> 3, d-1))
		)...
	]
	for I in CartesianIndices(chunks)
		for J in neighbours
			K = I + J
			# Not needed because periodic boundaries: !all(1 .≤ Tuple(K) .≤ chunks) && continue
			K = CartesianIndex(mod1.(Tuple(K) .- 1, chunks))
			push!(neighbour_pairs, (I, K))
		end
	end
	return neighbour_pairs
end


function chunked_convolution_size(
	grid::AbstractArray{<: AbstractVector{<: Integer}, N},
	neighbour_pairs::Vector{<: NTuple{2, CartesianIndex{N}}}
) where N
	len = 0
	for (p, (p1, p2)) in enumerate(neighbour_pairs)
		i = grid[p1]
		j = grid[p2]
		len += length(i) * length(j)
	end
	num = 0
	for i in grid
		num += length(i)
	end
	return len, num
end

#function testunique(a::Vector{NTuple{2, CartesianIndex{N}}}) where N  hallo hallo                                                                                         for i in eachindex(a)                                                                                                                                         for j = i+1:length(a)                                                                                                                                         if (a[i][1] == a[j][1] && a[i][2] == a[j][2]) || (a[i][1] == a[j][2] && a[i][2] == a[j][1])                                                                   error()                                                                                                                                                       end                                                                                                                                                           end                                                                                                                                                           end                                                                                                                                                           end


function compute_chunked_convolution(
	kernel::Function,
	kernel_radius::Real,
	shape::NTuple{N, Integer},
	samples::AbstractMatrix{<: Real},
	grid::AbstractArray{<: AbstractVector{<: Integer}, N},
	neighbour_pairs::Vector{<: NTuple{2, CartesianIndex{N}}}
) where N
	convolution = Vector{Matrix{Float64}}(undef, length(neighbour_pairs))
	Δ = Matrix{Float64}(undef, N, Threads.nthreads())
	for p in eachindex(neighbour_pairs) # Can't use enumerate because doesn't work with @threads
		p1, p2 = neighbour_pairs[p]
		i = grid[p1]
		j = grid[p2]
		c = Matrix{Float64}(undef, length(i), length(j))
		@views @inbounds Threads.@threads for I in CartesianIndices(c) # n in axes(c, 2), m in axes(c, 1)
			Δthr = Δ[:, Threads.threadid()]
			m, n = Tuple(I)
			ii = i[m]
			jj = j[n]
			# Take care of "periodic" distance
			for d = 1:N
				δ = abs(samples[d, ii] - samples[d, jj])
				if δ > π/2
					δ -= π/2
				end
				Δthr[d] = δ
			end
			@. Δthr = (shape * Δthr)^2
			kr = sqrt(sum(Δthr))
			@inbounds if kr > kernel_radius
				c[m, n] = 0
			else
				c[m, n] = kernel(kr)
			end
		end
		@inbounds convolution[p] = c
		GC.safepoint()
	end
	return convolution
end

function prepare_efficient_grid(grid::AbstractArray{<: AbstractVector{<: Integer}, N}) where N
	efficient_grid = Array{NTuple{2, Int}, N}(undef, size(grid))
	l = 1
	maxlength = 0
	for i in eachindex(grid)
		m = length(grid[i])
		if m > maxlength
			maxlength = m
		end
		k = l + m
		efficient_grid[i] = (l, k-1)
		l = k
	end

	return efficient_grid, maxlength
end

# Applied to permuted x,y!, use reverse_chunked_permutation() to get back to original ordering
function apply_chunked_convolution!(
	tmp::AbstractVector{<: T}, # length(tmp) == maximum.(length, grid)
	y::AbstractVector{<: T},
	x::AbstractVector{<: T},
	grid::AbstractArray{<: NTuple{2, Integer}, N},
	neighbour_pairs::Vector{<: NTuple{2, CartesianIndex{N}}},
	convolution::AbstractVector{<: AbstractMatrix{<: Real}}
) where {T <: Real, N}
	@views for (p, (p1, p2)) in enumerate(neighbour_pairs)
		l1, u1 = grid[p1]
		l2, u2 = grid[p2]
		(l1 > u1 || l2 > u2) && continue
		v1 = x[l1:u1]
		v2 = x[l2:u2]
		c = convolution[p]
		mul!(tmp[1:size(c, 1)], c, v2)
		y[l1:u1] .+= tmp[1:size(c, 1)]
		p1 == p2 && continue
		mul!(tmp[1:size(c, 2)], c', v1)
		y[l2:u2] .+= tmp[1:size(c, 2)]
	end
	return y
end



function chunked_permutation!(
	y::AbstractVector{<: Real},
	x::AbstractVector{<: Real},
	grid::AbstractArray{<: AbstractVector{<: Integer}, N}
) where N
	@assert length(y) == length(x)
	j = 1
	@views for i in eachindex(grid)
		idx = grid[i]
		k = j+length(idx)
		y[j:k-1] = x[idx]
		j = k
	end
	return y
end
function inv_chunked_permutation!(
	y::AbstractVector{<: Real},
	x::AbstractVector{<: Real},
	grid::AbstractArray{<: AbstractVector{<: Integer}, N}
) where N
	@assert length(y) == length(x)
	j = 1
	@views for i in eachindex(grid)
		idx = grid[i]
		k = j+length(idx)
		y[idx] = x[j:k-1]
		j = k
	end
	return y
end


# Need to use P to permute a vector, then use C as often as you like, then use P' to invert permutation
function plan_chunked_convolution(
	kernel::Function,
	kernel_radius::Real,
	shape::NTuple{N, Integer}, # FOV
	samples::AbstractMatrix{<: T},
	chunks::NTuple{N, Integer}
) where {T <: Real, N}
	
	@assert size(samples, 1) == N # TODO: question is whether in general it is better to do samples = Vector{NTuple{N, Float64}}(...)? What are advantages/disadvantages?
	# adv: dim is checked at compile time, could just use length(samples) instead size(samples, 2), it's easier to understand since just a list of tuples.
	# dis: harder to access only first dimension, need to use array comprehension.
	# To check what's best, need to collect more examples from all the code

	grid = divide_into_chunks(samples, chunks)

	efficient_grid, maxlength = prepare_efficient_grid(grid)

	neighbour_pairs = generate_neighbour_pairs(chunks)

	@show chunked_convolution_size(grid, neighbour_pairs)
	error()

	convolution = compute_chunked_convolution(kernel, kernel_radius, shape, samples, grid, neighbour_pairs)
	tmp = Vector{T}(undef, maxlength)

	C = HermitianOperator{Float64}(
		size(samples, 2),
		(y, x) -> begin
			turbo_wipe!(y)
			apply_chunked_convolution!(tmp, y, x, efficient_grid, neighbour_pairs, convolution)
			y
		end;
		out=Vector{Float64}(undef, 0)
	)

	P = UnitaryOperator{Float64}(
		size(samples, 2),
		( (y, x) -> chunked_permutation!(y, x, grid) );
		adj=( (y, x) -> inv_chunked_permutation!(y, x, grid) ),
		out=Vector{Float64}(undef, 0)
	)

	return C, P
end


# TODO: evaluate kernel prior to computing the convolution, then linearly interpolate
# k goes from -pi to pi here
# r is the FOV in indices because δx = 2π / Δk = 2π / 2π = 1
function pipe_kernel(kr::Real, N::Integer)
	if kr != 0
		return besselj(0.5N, 0.5kr)^2 / kr^N # TODO: outsource the 0.5 in front of kr? makes code more complicated but could save time, how much?
	else
		return 0.25^N / SpecialFunctions.gamma(1 + 0.5N)^2
	end
end

function pipe_density_correction(
	samples::AbstractMatrix{<: Real}, # [-π, π]
	shape::NTuple{N, Integer};
	tol::Real=1e-4,
	maxiter::Integer=30,
	kernel_zeros::Integer=2
) where N

	# TODO: uncomment @assert maxiter > 0
	#=
		From the kernel radius, the number of chunks can be determined
		TODO: pi is gone

		π / 2π sqrt(sum(abs2, k .* shape)) < besselzero
		sqrt(sum(abs2, k .* shape)) < 2 * besselzero
		sum(abs2, k .* shape) < (2 * besselzero)^2

		for my k ∈ [-π, π]

		along one direction
		ki * shape_i < 2 * besselzero
		ki < 2 * besselzero / shape_i

		So, the number of chunks in that direction is
		2π / (2 * besselzero / shape_i)
		π * shape_i / besselzero
	=#

	kernel(kr) = pipe_kernel(kr, N)

	@show kernel_radius = 2 * besselj_zero(1, kernel_zeros) # Times two because we use *0.5 in the kernel argument, and this is for k ∈ [-π, π] and r = shape
	@show kernel_norm = kernel_integral(kernel, N, kernel_radius)

	@show chunks = ceil.(Int, 2π .* shape ./ kernel_radius)

	C, P = plan_chunked_convolution(kernel, kernel_radius, shape, samples, chunks)

	weights = Vector{Float64}(undef, size(C, 2))
	density = Vector{Float64}(undef, size(C, 2))
	#tmp = ones(size(C, 2))
	#tmp = sqrt.(dropdims(sum(abs2, samples; dims=1); dims=1)) ./ π
	tmp = ones(size(C, 2))
	#tmp = zeros(size(C, 2))
	#tmp[200 * 400 + 200] = 1

	mul!(weights, P, tmp)

	iter = 0
	err = Inf
	while (err > tol && iter < maxiter)
		@show iter
		mul!(density, C, weights)
		#TODO uncomment: any(iszero, density) && error("density estimate produced zero")
		weights ./= density
		err = sqrt(sum(abs2, density .- 1) ./ length(density))
		iter += 1
	end

	err > tol && @warn "Pipe density compensation didn't converge"

	mul!(tmp, P', weights)
	tmp2 = weights
	weights = tmp
	weights .*= kernel_norm
	mul!(tmp2, P', density)
	density = tmp2

	return weights, density, err
end

