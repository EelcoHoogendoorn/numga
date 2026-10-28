"""The nested E6, E7 and E8 root systems and their invariant rotation planes."""

from functools import reduce
from operator import mul

import numpy as np
import pytest

from numga import stack

from examples.math.e8.core import mv
from examples.math.e_series import core, scenarios


@pytest.mark.parametrize(
    "case_index, n_roots, rates",
    [(0, 72, [1, 4, 5]), (1, 126, [1, 5, 7]), (2, 240, [1, 7, 11, 13])],
)
def test_root_closure_and_rotation_planes(case_index: int, n_roots: int, rates: list[int]):
    n_rank = scenarios.ranks[case_index]
    n_steps = scenarios.periods[case_index]
    action, roots, plane_x, plane_y, angles = scenarios.family()[case_index]
    simple = core.simple_roots()[:n_rank]
    tolerance = 1e-10

    span = (roots * (roots | core.Vector)).sum(axis=(0, 1)) / (2 * n_steps)
    plane_span = (plane_x * (plane_x | core.Vector)
                  + plane_y * (plane_y | core.Vector)).sum(axis=0)
    remaining = span - plane_span
    frame = stack((plane_x, plane_y), axis=1).reshape(-1)

    # Independent planar turns reconstruct the motion; E7 also reverses one direction.
    factors = (-(plane_x ^ plane_y) * angles / 2).exp()
    turning = reduce(mul, factors)
    reconstructed = (turning >> core.Vector)(core.Vector - 2 * remaining)

    # checks
    coefficients = roots.kernel.reshape(-1, 8)
    assert len(coefficients) == n_roots
    distances = np.linalg.norm(coefficients[:, None] - coefficients[None], axis=-1)
    np.fill_diagonal(distances, np.inf)
    assert distances.min() > 1
    np.testing.assert_allclose(roots.scalar_norm_squared().to_array(), 2, atol=tolerance)

    # Each generating reflection permutes the complete set, including negative roots.
    reflected = simple.normalized()[:, None, None] >> roots[None]
    distances = np.linalg.norm(reflected.kernel.reshape(-1, 8)[:, None]
                               - coefficients[None], axis=-1)
    np.testing.assert_allclose(distances.min(axis=-1), 0, atol=tolerance)
    np.testing.assert_allclose((action(roots[-1]) - roots[0]).kernel, 0, atol=tolerance)
    displacements = (roots[1:] - roots[0]).scalar_norm_squared().to_array()
    assert np.all(displacements.sum(axis=-1) > 1)

    np.testing.assert_allclose((span(span) - span).kernel, 0, atol=tolerance)
    np.testing.assert_allclose((span(roots) - roots).kernel, 0, atol=tolerance)
    np.testing.assert_allclose(span.trace().to_array(), n_rank, atol=tolerance)
    np.testing.assert_allclose((frame[:, None] | frame[None]).to_array(),
                               np.eye(len(frame)), atol=tolerance)
    np.testing.assert_allclose(angles.to_array() * n_steps / (2 * np.pi), rates, atol=tolerance)
    np.testing.assert_allclose((action(plane_x) - plane_x * angles.cos()
                               - plane_y * angles.sin()).kernel, 0, atol=tolerance)
    np.testing.assert_allclose((action(plane_y) + plane_x * angles.sin()
                               - plane_y * angles.cos()).kernel, 0, atol=tolerance)
    np.testing.assert_allclose(remaining.trace().to_array(), n_rank % 2, atol=tolerance)
    np.testing.assert_allclose((remaining(remaining) - remaining).kernel, 0, atol=tolerance)
    np.testing.assert_allclose((action(remaining) + remaining).kernel, 0, atol=tolerance)
    np.testing.assert_allclose((reconstructed - action).kernel, 0, atol=tolerance)


def test_nested_root_systems_and_mirror_covariance():
    family = scenarios.family()
    mirror = (mv.a + 2 * mv.b - 3 * mv.c + mv.h).normalized()
    spans = tuple((roots * (roots | core.Vector)).sum(axis=(0, 1)) / (2 * n_steps)
                  for (_, roots, _, _, _), n_steps in zip(family, scenarios.periods))
    tolerance = 1e-10

    # checks
    for inner, outer, inner_span, outer_span in zip(family[:-1], family[1:], spans[:-1], spans[1:]):
        inner_roots = inner[1].kernel.reshape(-1, 8)
        outer_roots = outer[1].kernel.reshape(-1, 8)
        distances = np.linalg.norm(inner_roots[:, None] - outer_roots[None], axis=-1)
        np.testing.assert_allclose(distances.min(axis=-1), 0, atol=tolerance)
        np.testing.assert_allclose((outer_span(inner_span) - inner_span).kernel, 0, atol=tolerance)
        np.testing.assert_allclose((inner_span(outer_span) - inner_span).kernel, 0, atol=tolerance)

    for n_rank, n_steps, case, span in zip(scenarios.ranks, scenarios.periods, family, spans):
        action, roots, _, _, _ = case
        reflected_action, reflected_roots = core.root_system(mirror >> core.simple_roots()[:n_rank], n_steps)
        reflected_span = (reflected_roots * (reflected_roots | core.Vector)).sum(axis=(0, 1)) / (2 * n_steps)
        np.testing.assert_allclose((reflected_roots - (mirror >> roots)).kernel, 0, atol=tolerance)
        np.testing.assert_allclose((reflected_action - (mirror >> action(mirror << core.Vector))).kernel,
                                   0, atol=tolerance)
        np.testing.assert_allclose((reflected_span - (mirror >> span(mirror << core.Vector))).kernel,
                                   0, atol=tolerance)
