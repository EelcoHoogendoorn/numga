"""The nested E6, E7 and E8 root systems, and their covariance under a mirror."""

import numpy as np

from examples.math.e8.core import mv
from examples.math.e_series import core, render, scenarios


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


def test_the_diagram_and_the_nested_roots_draw():
    cases = scenarios.family()
    _, _, plane_x, plane_y, _ = cases[-1]
    render.draw_diagram(core.simple_roots(), scenarios.ranks, scenarios.names)
    render.draw_nested_roots(tuple(roots for _, roots, _, _, _ in cases), plane_x[0], plane_y[0], scenarios.names)
