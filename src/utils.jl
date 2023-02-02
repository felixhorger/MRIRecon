
# Utilised by functions planning a LinearOperator
function check_allocate(
	a::Array{<: T, N},
	shape::NTuple{N, Integer}
) where {T, N}
	if length(a) == 0
		a = Array{T, N}(undef, shape)
	else
		shape_a = size(a)
		@assert all(shape_a .== shape) "Expected array of shape $shape, but got $shape_a."
	end
	return a
end
check_allocate(a, shape::Integer...) = check_allocate(a, shape)

empty(A::Type{<: AbstractArray{T, N}}) where {T, N} = Array{T, N}(undef, ntuple(_ -> 0, N))

