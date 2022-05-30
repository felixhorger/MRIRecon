
"""
	plan_PSF(F::LinearMap, M::LinearMap [, S::LinearMap])

"""
plan_psf(F::LinearMap, M::LinearMap) = F' * M * F
plan_psf(F::LinearMap, M::LinearMap, S::LinearMap) = S' * F' * M * F * S

