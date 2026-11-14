"""The Toda vacuum spectrum and cubic couplings agree with E8's projected root geometry."""

import numpy as np

from examples.quantum.affine_toda import core, render, scenarios


def test_all_cubic_couplings_match_root_triangles():
    roots, marks, _, _, masses, modes, _, radii, *_ = scenarios.spectrum()
    species, triangles, closure, couplings, predicted = scenarios.interactions()
    tolerance = 1e-8
    allowed = closure < tolerance

    # checks
    assert (core.potential(0 * modes[0], roots, marks).abs() < tolerance).all()
    assert ((radii / radii[0] - masses / masses[0]).abs() < tolerance).all()
    assert len(species) == 120
    assert np.count_nonzero(allowed) == 56
    np.testing.assert_array_equal(allowed, couplings.abs() > tolerance)
    assert (triangles[allowed].sum(axis=1).scalar_norm_squared() < tolerance).all()
    assert ((couplings[allowed].abs() - predicted[allowed]).abs() < tolerance).all()


def test_the_masses_and_triangles_draw():
    *_, roots, _, horizontal, vertical = scenarios.spectrum()
    triangles, _ = core.root_triangles(roots, scenarios.example_species[:1])
    render.draw_roots(roots, horizontal, vertical)
    render.triangles_figure(triangles, roots, scenarios.example_species[:1], horizontal, vertical)
