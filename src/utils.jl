
# Utilised by functions planning a LinearOperator
function check_allocate(
	a::AbstractArray{T, N},
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

# FakeArray from FFTW
Base.show(io::IO, a::FakeArray) = print(io, "FakeArray")
Base.show(io::IO, m::MIME"text/plain", a::FakeArray) = show(io, a)
check_allocate(a::FakeArray{T, N}, shape::NTuple{N, Integer}) where {T, N} = empty(Array{T, N})
FakeArray{T, 1}(num::Integer) where T = FakeArray{T, 1}((num,), (1,))
FakeArray{T, N}(shape::NTuple{N, Integer}) where {T, N} = FakeArray{T, N}(shape, cumprod((1, shape[1:N-1]...)))

empty(A::Type{<: AbstractArray{T, N}}) where {T, N} = Array{T, N}(undef, ntuple(_ -> 0, N))

