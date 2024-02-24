using Revise
import MRIRecon
import MRITrajectories
import SpecialFunctions
using FFTW
import PyPlot as plt


shape = (200, 200)
k = reshape(MRITrajectories.radial_spokes(150, 200), 2, :)
extrema(k)

# Test Jinc

centre = first.(MRIRecon.centre_indices.(shape, 1))


kernel_radius = 2 * MRIRecon.besselj_zero(1, 3)
MRIRecon.kernel_integral(kr -> MRIRecon.pipe_kernel.(kr, 2), kernel_radius) ./ 2π
TODO

kr = range(0, kernel_radius; length=1000)
plt.cla()
plt.plot(MRIRecon.pipe_kernel.(kr, 2))
plt.plot(MRIRecon.Bessels.besselj.(2/2, kr./2).^2 ./ kr.^2)
plt.plot(MRIRecon.pipe_kernel.(kr, 3))
plt.plot(MRIRecon.Bessels.besselj.(3/2, kr./2).^2 ./ kr.^3)
plt.grid()


kerneleval = Array{Float64, 2}(undef, shape)
kerneleval_cut = Array{Float64, 2}(undef, shape)
for I in CartesianIndices(shape)
	kr = sqrt(sum(abs2, Tuple(I) .- centre))
	v = MRIRecon.jinc(kr)
	kerneleval[I] = v
	if kr > kernel_radius
		kerneleval_cut[I] = 0.0
	else
		kerneleval_cut[I] = v
	end
end

fig, axs = plt.subplots(2, 2)
axs[1, 1].imshow(abs.(kerneleval))
axs[1, 2].imshow(abs.(kerneleval_cut))
axs[2, 1].imshow(fftshift(abs.(ifft(ifftshift(kerneleval .^2)))))
axs[2, 2].imshow(fftshift(abs.(ifft(ifftshift(kerneleval_cut .^2 )))))

plt.figure()
plt.imshow(abs.(kerneleval_cut))

mshape = (10, 10)
n = MRIRecon.generate_neighbour_pairs(mshape)
length(n)
foreach(println, n)

b = zeros(mshape)
for (p, q) in n
	b[p] += 1
	p == q && continue
	b[q] += 1
end
b


kx = repeat(range(-π, π; length=400+1)[1:end-1], inner=400)
ky = repeat(range(-π, π; length=400+1)[1:end-1], 400)
k = vcat(reshape.((kx, ky), 1, :)...)

w, d, err = MRIRecon.pipe_density_correction(k, shape; tol=1e-6, maxiter=30, kernel_zeros=2);

plt.cla()
plt.plot(reshape(d, 400, 400)[:, 201]')

plt.cla()
plt.plot(reshape(w, 200, 150)[:, 10]')

plt.cla()
plt.imshow(sqrt.(reshape(d, 64, 96)); vmin=0.9, vmax=1.1)

plt.cla()
plt.scatter(k[1, :], k[2, :], s=5, c=w)

plt.cla()
plt.plot(reshape(d, 64, 96)[:, 1]')


F = MRIRecon.plan_fourier_transform(k, (shape..., 1))
a = zeros(ComplexF64, shape)
a[20:40, 20:40] .= 1
y = copy(F' * (w .* (F * vec(a))))

fig, axs = plt.subplots(1, 2)
axs[1].imshow(abs.(a))
axs[2].imshow(abs.(reshape(y, shape)))




kgrid = floor.(Int, (k ./ 2π .+ 0.5) .* shape .+ 1)
a = zeros(shape)
@views for i in axes(kgrid, 2)
	a[Tuple(kgrid[:, i])...] = w[i]
end
plt.cla()
#plt.plot(d)
plt.imshow(a)

