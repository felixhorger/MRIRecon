
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

