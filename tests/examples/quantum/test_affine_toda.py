"""The Toda vacuum spectrum and cubic response agree with E8's projected root geometry."""

import numpy as np

from examples.quantum.affine_toda import core, scenarios


def test_potential_vacuum_and_derivatives():
    interaction_roots, marks, curvature, cubic, masses, modes, _, _, _ = scenarios.spectrum()
    tolerance = 1e-10
    direction = modes[0] + 0.6 * modes[1] - 0.4 * modes[2]
    step = 0.005
    amplitudes = step * np.array([-2, -1, 1, 2])
    energies = core.potential(amplitudes * direction, interaction_roots, marks)
    second_derivative = (energies[2] + energies[1]) / step**2
    third_derivative = (energies[3] - 2 * energies[2] + 2 * energies[1] - energies[0]) / (2 * step**3)

    # checks
    np.testing.assert_allclose((marks * interaction_roots).sum(axis=0).kernel, 0, atol=tolerance)
    np.testing.assert_allclose(interaction_roots.scalar_norm_squared().to_array(), 2, atol=tolerance)
    np.testing.assert_allclose(core.potential(0 * direction, interaction_roots, marks).to_array(), 0, atol=tolerance)
    assert np.all(masses > 0)
    # The modes diagonalize the curvature, with the squared masses on its diagonal.
    np.testing.assert_allclose(curvature(modes[:, None], modes[None, :]).to_array(), np.diag(masses.squared().to_array()), atol=tolerance)
    np.testing.assert_allclose(second_derivative.to_array(), curvature(direction, direction).to_array(), atol=1e-5)
    np.testing.assert_allclose(third_derivative.to_array(), cubic(direction, direction, direction).to_array(), atol=1e-5)


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
