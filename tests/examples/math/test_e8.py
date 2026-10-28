"""E8's roots close under their mirrors and split into four commuting rotations."""

from functools import reduce
from operator import mul

import numpy as np

from numga.extensor import stack

from examples.math.e8 import core, scenarios


def test_roots_and_four_plane_rotation():
    roots, plane_x, plane_y, planes, rates, coxeter, seeds = scenarios.root_system()
    simple_roots = core.simple_roots()
    n_steps = core.coxeter_order
    tolerance = 1e-10

    # checks
    coefficients = roots.kernel.reshape(-1, 8)
    assert len(np.unique(np.round(coefficients, decimals=9), axis=0)) == 240
    np.testing.assert_allclose(roots.scalar_norm_squared().to_array(), 2, atol=tolerance)
    np.testing.assert_allclose(((coxeter >> roots[-1]) - seeds).kernel, 0, atol=tolerance)

    # Every generating mirror permutes all 240 roots, including their signs.
    reflected = simple_roots.normalized()[:, None, None] >> roots[None]
    reflected_coefficients = reflected.kernel.reshape(-1, 8)
    distances = np.linalg.norm(reflected_coefficients[:, None] - coefficients[None], axis=-1)
    np.testing.assert_allclose(distances.min(axis=-1), 0, atol=tolerance)

    frame = stack((plane_x, plane_y), axis=1).reshape(-1)
    np.testing.assert_allclose((frame[:, None] | frame[None]).to_array(), np.eye(8), atol=tolerance)
    np.testing.assert_allclose(rates.to_array(), [1, 7, 11, 13], atol=tolerance)
    np.testing.assert_allclose(planes.commutator(planes[:, None]).kernel, 0, atol=tolerance)
    np.testing.assert_allclose((planes ^ planes).kernel, 0, atol=tolerance)

    factors = (-planes * rates * np.pi / n_steps).exp()
    factored = reduce(mul, factors)
    np.testing.assert_allclose(((factored >> core.Vector) - (coxeter >> core.Vector)).kernel, 0, atol=tolerance)

    # A bounded exponential advances the animation; its samples recover every discrete orbit.
    n_substeps = 6
    generator = -(planes * rates).sum(axis=0) * np.pi / n_steps
    advance = (generator / n_substeps).exp() >> core.Vector
    interpolated = stack(core.orbit(advance, seeds, n_steps * n_substeps + 1))
    np.testing.assert_allclose((interpolated[:-1:n_substeps] - roots).kernel, 0, atol=tolerance)
    np.testing.assert_allclose((interpolated[-1] - seeds).kernel, 0, atol=tolerance)
