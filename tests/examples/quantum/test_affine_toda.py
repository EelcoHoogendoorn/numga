"""The Toda vacuum spectrum and cubic response agree with E8's projected root geometry."""

import numpy as np

from examples.quantum.affine_toda import render, scenarios


def test_all_cubic_couplings_match_root_triangles():
    _, _, _, _, masses, _, roots, plane_x, plane_y = scenarios.spectrum()
    species, triangles, closure, couplings, areas = scenarios.interactions()
    tolerance = 1e-10
    radii = (roots[0] | (plane_x ^ plane_y)).scalar_norm_squared().square_root()
    allowed = closure < tolerance
    mass_areas = areas * (masses[0] / radii[0]).squared()

    # checks
    np.testing.assert_allclose((radii / radii[0] - masses / masses[0]).to_array(), 0, atol=tolerance)
    assert len(species) == 120
    assert np.count_nonzero(allowed) == 56
    np.testing.assert_array_equal(allowed, couplings.abs() > tolerance)
    np.testing.assert_allclose(triangles[allowed].sum(axis=1).kernel, 0, atol=tolerance)
    np.testing.assert_allclose((couplings[allowed].abs() - 4 / np.sqrt(30) * mass_areas[allowed]).to_array(), 0, atol=tolerance)


def test_the_masses_draw():
    *_, roots, horizontal, vertical = scenarios.spectrum()
    render.draw_roots(roots, horizontal, vertical)
