
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

