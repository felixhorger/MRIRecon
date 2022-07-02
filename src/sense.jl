
"""
	plan_PSF([S::LinearMap,] F::LinearMap, M::LinearMap)
	Matrices in the order in which they are applied

"""
plan_psf(F::LinearMap, M::LinearMap) = F' * M * F
plan_psf(S::LinearMap, F::LinearMap, M::LinearMap) = S' * F' * M * F * S

