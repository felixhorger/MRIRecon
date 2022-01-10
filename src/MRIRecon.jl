
module MRIRecon

	function root_sum_of_squares(signals::AbstractArray{<: T}, dim::Val{N}=Val(1)) where T <: Number where N
		result = dropdims(sum(abs2, signals; dims=N); dims=N)
		@. result = sqrt(result)
	end

end

