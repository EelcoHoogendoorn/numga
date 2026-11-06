"""E8's roots, the four planes of its Coxeter rotation and their winding rates, and their figures."""

import numpy as np

from examples.math.e8 import render, scenarios


def test_the_root_system_turns_its_planes_at_their_rates_and_draws():
    roots, plane_x, plane_y, _, rates, _, _ = scenarios.root_system()
    np.testing.assert_allclose(rates.to_array(), [1, 7, 11, 13], atol=1e-6)
    render.draw_roots(roots, plane_x[0], plane_y[0])
    # The roots' own Coxeter steps, as the frames.
    next(render.animate_planes(roots, roots, plane_x, plane_y))
