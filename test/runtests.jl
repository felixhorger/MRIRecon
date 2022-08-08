
# Sorry, the quality of commenting/documentation is low
#using Revise
#using ImageView
#import PyPlot as plt
using Test
using IterativeSolvers
using Statistics
using FFTW
import HDF5
import MRIPhantoms
import MRIRecon


function run_espirit_python()
	# Run python version
	if !isdir("espirit_python")
		run(`git clone https://github.com/mikgroup/espirit-python`)
		mv("espirit-python", "espirit_python") # Argggg
		touch("espirit_python/__init__.py") 
	else
		# TODO: check version of existing repo
	end
	if !isfile("espirit_test.h5")
		run(`python3 espirit_test.py`)
	else
		# TODO: test if file is newer than the git repo
	end
	# Read testset
	data, py_sensitivities = HDF5.h5open("espirit_test.h5", "r") do f
		data = read(f, "data")
		data = permutedims(data, (3, 2, 1))
		py_sensitivities = read(f, "coil_sensitivities")
		py_sensitivities = permutedims(py_sensitivities, (3, 2, 1))
		data, py_sensitivities
	end
	# Reconstruct
	shape = (size(data, 1), size(data, 2))
	calibration_area = (24, 24)
	kernelsize = (6, 6)
	centre = MRIRecon.centre_indices.(shape, calibration_area)
	σ_cut = 0.01
	λ_cut = 0.9925
	sensitivities = MRIRecon.estimate_sensitivities(Val(:ESPIRIT), data[centre..., :], shape, kernelsize, σ_cut, λ_cut)
	recon = ifft(ifftshift(data, (1, 2)), (1, 2))
	projection = sensitivities .* sum(recon .* conj.(sensitivities); dims=3)
	return sensitivities, py_sensitivities, recon, projection
end

# ESPIRiT
rel_diff_recon_projection, rel_diff_to_python = let
	(sensitivities, py_sensitivities, recon, projection) = run_espirit_python()
	# Leave all the plots in here in case of and issue
	#imshow(abs.(sensitivities))
	#imshow(abs.(py_sensitivities))
	# Test validity of sensitivities, see Uecker2013, Eq. 20 and Fig. 5
	rel_diff_recon_projection = @. abs(projection - recon) / $maximum(abs(recon))
	#vmin, vmax = extrema(abs.(recon))
	#plt.figure()
	#plt.imshow(abs.(projection .- recon)[:, :, 1]; vmin, vmax)
	# Compare against the python implementation
	# Shift
	sensitivities = fftshift(sensitivities, 1:2)
	# Compute absolute values, masking according to the python version
	# this is necessary since this here implements a smooth transition, see Uecker2013a
	zero_indices = py_sensitivities .== 0
	abs_sensitivities = abs.(sensitivities)
	abs_sensitivities[zero_indices] .= 1
	abs_py_sensitivities = abs.(py_sensitivities)
	abs_py_sensitivities[zero_indices] .= 1
	rel_diff_to_python = vec(@. 2 * abs(abs_sensitivities - abs_py_sensitivities) / (abs_sensitivities + abs_py_sensitivities))
	# Return
	rel_diff_recon_projection, rel_diff_to_python
end
@test all(rel_diff_recon_projection .< 5e-2)
@test 3e-2 > quantile(rel_diff_to_python, 0.99)



function get_phantom()
	# Dimensions
	num_channels = 16
	phase_encoding_shape = (126, 124)
	shape = (128, phase_encoding_shape...)

	# Sensitivities
	sensitivities = MRIPhantoms.coil_sensitivities(shape, (8, 2), 0.3)

	# Operators and sampling
	S = MRIRecon.plan_sensitivities(sensitivities, 1)
	calibration_area = (24, 24)
	calibration_indices = MRIRecon.centre_indices.(phase_encoding_shape, calibration_area)
	sampling_indices = [
		[CartesianIndex(line, partition) for line = 1:2:shape[2] for partition = 1:shape[3]];
		[CartesianIndex(line, partition) for line in calibration_indices[1] for partition in calibration_indices[2]]
	]
	fftshifted_indices = MRIRecon.fftshift(sampling_indices, shape[2:3])
	M = MRIRecon.plan_masking(fftshifted_indices, (shape..., num_channels, 1))
	F = MRIRecon.plan_spatial_ft(Array{ComplexF64, 5}(undef, shape..., num_channels, 1), 1:3)
	A = MRIRecon.plan_psf(M, F, S)
	GC.gc(true)

	# Measure phantom
	upsampling = (2, 2, 2)
	phantom, _ = MRIPhantoms.homogeneous(shape, upsampling)
	kspace = fft(sensitivities .* phantom, 1:3)
	backprojection = S' * F' * M * vec(kspace)
	phantom_backprojection = A * vec(phantom)

	return (
		phantom,
		sensitivities,
		fftshift(kspace, 1:3),
		backprojection,
		phantom_backprojection,
		calibration_indices,
		sampling_indices,
		fftshifted_indices,
		M, F, A
	)
end

function run_espirit(kspace, calibration_indices)
	kspace = ifft(ifftshift(kspace, 1), 1)
	# Dimensions
	num_channels = size(kspace, 4)
	shape = (size(kspace, 2), size(kspace, 3))
	num = size(kspace, 1)
	kernelsize = (6, 6)
	σ_cut = 0.001
	λ_cut = 0.99
	# Iterate along readout
	sensitivities = Array{ComplexF64, 4}(undef, num, shape..., num_channels)
	for r = 1:num
		@views sensitivities[r, :, :, :] = MRIRecon.estimate_sensitivities(
			Val(:ESPIRIT),
			kspace[r, calibration_indices..., :],
			shape,
			kernelsize,
			σ_cut, λ_cut
		)
	end
	return sensitivities
end

function reconstruct_from_espirit(sensitivities, backprojection, M, F)
	S = MRIRecon.plan_sensitivities(sensitivities, 1)
	A = MRIRecon.plan_psf(M, F, S)
	recon = reshape(cg(A, backprojection, maxiter=30), size(sensitivities)[1:3])
	return recon
end

function run_coil_compression(kspace, backprojection, sampling_indices, calibration_indices, fftshifted_indices, A)
	# Permute dims and flatten so that the data is ordered like in typical MRI raw data
	# kspace[readout frequency, spatial dims..., channels]
	sampled_kspace = @view kspace[:, sampling_indices, :]
	sampled_kspace = permutedims(sampled_kspace, (1, 3, 2))
	# sampled_kspace[readout frequency, channels, spatial dims]
	# Here, the kspace has the same format as real MRI data
	# Fourier transform along readout
	sampled_kspace = ifft(ifftshift(sampled_kspace, 1), 1)
	GC.gc(true)

	# Compute coil compression matrices
	C, σ = MRIRecon.channel_compression_matrix(sampled_kspace)
	# Note: In the non-Cartesian case, I guess everything has to go into one array
	C = MRIRecon.cut_virtual_channels(C, σ, 5e-3)
	C_aligned = MRIRecon.align_channels(C)
	num_virtual_channels = size(C, 1)
	# Compress kspace
	compressed_kspace = MRIRecon.apply_channel_compression(sampled_kspace, C_aligned);
	# compressed_kspace[readout, channels, other spatial dims]

	# Put in dense matrix
	shape = (size(kspace, 2), size(kspace, 3))
	compressed_kspace = MRIRecon.sparse2dense(compressed_kspace, sampling_indices, shape)
	compressed_kspace = permutedims(compressed_kspace, (1, 3, 4, 2)) # Put channels last

	kernelsize = (6, 6)
	σ_cut = 0.001
	λ_cut = 0.99
	compressed_sensitivities = Array{ComplexF64, 4}(undef, size(kspace, 1), shape..., num_virtual_channels)
	for r = 1:size(kspace, 1)
		@views compressed_sensitivities[r, :, :, :] = MRIRecon.estimate_sensitivities(
			Val(:ESPIRIT),
			compressed_kspace[r, calibration_indices..., :],
			shape,
			kernelsize,
			σ_cut, λ_cut
		)
	end
	GC.gc(true)

	# Get operator for compressed sensitivities
	S_compressed = MRIRecon.plan_sensitivities(compressed_sensitivities, 1)
	M_compressed = MRIRecon.plan_masking(fftshifted_indices, (size(kspace, 1), shape..., num_virtual_channels, 1))
	F_compressed = MRIRecon.plan_spatial_ft(Array{ComplexF64, 5}(undef, size(kspace, 1), shape..., num_virtual_channels, 1), 2:3)
	A_compressed = MRIRecon.plan_psf(M_compressed, F_compressed, S_compressed)

	# fftshift and backproject
	shifted_compressed_kspace = ifftshift(compressed_kspace, 2:3)
	compressed_backprojection = S_compressed' * F_compressed' * M_compressed * vec(shifted_compressed_kspace)

	# Run reconstructions, one with all channels, one with compressed channels
	recon = reshape(cg(A, backprojection, maxiter=10), size(kspace, 1), shape...)
	recon_compressed = reshape(cg(A_compressed, compressed_backprojection, maxiter=10), size(kspace, 1), shape...)
	return C, C_aligned, recon, recon_compressed
end

# Coil compression
let
	(
		phantom,
		sensitivities,
		kspace,
		backprojection,
		phantom_backprojection,
		calibration_indices,
		sampling_indices,
		fftshifted_indices,
		M, F, A
	) = get_phantom()
	@test backprojection ≈ phantom_backprojection

	sensitivities_uncompressed = run_espirit(kspace, calibration_indices)
	recon = reconstruct_from_espirit(sensitivities, backprojection, M, F)
	#imshow(abs.(permutedims(sensitivities .- sensitivities_uncompressed, (2, 3, 4, 1))))
	#imshow(abs.(permutedims(sensitivities, (2, 3, 4, 1))))
	#imshow(abs.(permutedims(sensitivities_uncompressed, (2, 3, 4, 1))))
	@test all(abs2.(phantom - recon) .< (2e-2)^2)
		
	C, C_aligned, recon, recon_compressed = run_coil_compression(kspace, backprojection, sampling_indices, calibration_indices, fftshifted_indices, A)
	@test sum(abs2, phantom .- recon) < sum(abs2, phantom .- recon_compressed)

	# Compute differences in "alignment"
	isaligned = true
	difference = Matrix{ComplexF64}(undef, size(C, 1), size(C, 2))
	for r = 2:size(C, 3)
		difference .= C[:, :, r] .- C[:, :, r-1]
		unaligned = sum(abs2, difference)
		difference .= C_aligned[:, :, r] .- C_aligned[:, :, r-1]
		aligned = sum(abs2, difference)
		if unaligned < aligned
			isaligned = false
			break
		end
	end
	@test isaligned
end

