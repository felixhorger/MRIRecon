
module MRIRecon

	#using FFTW
	using Base.Cartesian

	function root_sum_of_squares(signals::AbstractArray{<: Number}, dim::Val{N}=Val(1)) where N
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
	# TODO: Smooth high res scan for computing coil profiles? There is a paper on this

	@generated function half_fov_shift!(kspace::AbstractArray{<: Number, N}, axes::Val{M}) where {N,M}
		return quote
			@inbounds @nloops $N k kspace begin
				s = 0
				@nexprs $M (d -> s += k_d)
				if isodd(s)
					(@nref $N kspace k) *= -1
				end
			end
			return kspace
		end
	end
end

