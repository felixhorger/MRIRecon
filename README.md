MRIRecon.jl - Reconstruction Methods for Magnetic Resonance Imaging
===================================================================

You are probably looking for [MRIReco.jl](https://github.com/MagneticResonanceImaging/MRIReco.jl).
which is in the official Julia registry.

This package here is my humble approach to MRI Reconstruction.

If you use my package, please leave a star! Your appreciation is my motivation.
If my code helps you with your research, please cite it as follows:
```
	@misc{Horger2021,
		title = {Julia Package for MR Image Reconstruction (MRIRecon.jl)},
		url = {https://github.com/felixhorger/MRIRecon.jl},
		author = {Felix Horger},
		year = {2021}
	}
```


Aim
---

Provide generic algorithms for image reconstruction in MRI:

- Reconstruction Techniques
	- SENSE
	- GRAPPA
	- Partial Fourier
- Linear operators
	- (Non-)uniform Fourier transform
	- Masking (incl. sparse to dense transformation)
	- Coil sensitivities
	- Periodic convolution
- Coil combination
	- Root sum of squares
	- Roemer
	- Coil compression (and alignment)
- Sensitivity estimation
	- Filtered low-resolution reconstruction, normalising with root sum of squares
	- ESPIRIT (not functional atm and won't be for the time being)
- Fourier related utilities
	- Explicit Fourier matrix,
	- Evaluation of individual Fourier coefficients
	- FOV shifts
	- Fourier down/up sampling (Sinc-interpolation)
	- fftshift for indices

