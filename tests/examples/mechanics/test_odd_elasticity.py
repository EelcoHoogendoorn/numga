"""Work cycles, geometric covariance and overdamped motion in odd elastic lattices."""

from __future__ import annotations

import numpy as np
import pytest

from examples.mechanics.odd_elasticity import core
from numga.sparse import spdiag

mv, Displacement = core.mv, core.Displacement
x, y = core.directions


@pytest.fixture
def cycle() -> core.Displacement:
    n_samples = 48
    angles = np.linspace(0, 2 * np.pi, n_samples + 1)
    circle = (-mv.xy * angles / 2).exp() >> x
    stretch = 0.18 * x * (mv.x & Displacement) + 0.11 * y * (mv.y & Displacement)
    return stretch(circle) + 0.07 * x - 0.03 * y


def loop_work(response: core.Response, cycle: core.Displacement) -> core.Scalar:
    """The work of a linear force around a polygon, each segment's midpoint force paired with it."""
    return (response((cycle[1:] + cycle[:-1]) / 2) & (cycle[1:] - cycle[:-1])).sum(axis=0)


def test_cycle_work_is_oriented_area_and_changes_sign_on_reversal(cycle: core.Displacement):
    activity = 0.8
    direction = (x + 0.3 * y).normalized()
    response = core.bond_response(direction, mv.xy, 0.9, activity)
    passive = core.bond_response(direction, mv.xy, 0.9, 0)
    work = loop_work(response, cycle)
    area = (cycle[:-1] ^ cycle[1:]).sum(axis=0) / 2
    area_work = -activity * mv.xy.scalar_product(area)
    states = tuple(core.integrate_cycle(response, cycle))

    # checks
    np.testing.assert_allclose(work.kernel, area_work.kernel, atol=1e-12)
    np.testing.assert_allclose(loop_work(passive, cycle).kernel, 0, atol=1e-12)
    np.testing.assert_allclose(loop_work(response, cycle[::-1]).kernel, -work.kernel, atol=1e-12)
    np.testing.assert_allclose(states[-1][2].kernel, area_work.kernel, atol=1e-12)
    assert work.kernel.item() > 0.04


def test_mirror_transports_bond_force_and_preserves_cycle_work(cycle: core.Displacement):
    direction = (x + 0.4 * y).normalized()
    mirror = (mv.x - 0.2 * mv.y).normalized()
    response = core.bond_response(direction, mv.xy, 0.9, 0.8)
    reflected_response = core.bond_response(mirror >> direction, mirror >> mv.xy, 0.9, 0.8)
    reflected_cycle = mirror >> cycle

    # checks
    np.testing.assert_allclose(reflected_response(reflected_cycle).kernel, (mirror >> response(cycle)).kernel, atol=1e-12)
    np.testing.assert_allclose(loop_work(reflected_response, reflected_cycle).kernel,
                               loop_work(response, cycle).kernel, atol=1e-12)


def test_three_bond_families_pull_back_to_an_odd_shear_response(cycle: core.Displacement):
    stiffness, activity = 0.9, 0.8
    angles = np.arange(3) * np.pi / 3
    directions = (-mv.xy * angles / 2).exp() >> x
    response = core.bond_response(directions, mv.xy, stiffness, activity)
    shear_first = x * (mv.x & Displacement) - y * (mv.y & Displacement)
    shear_second = x * (mv.y & Displacement) + y * (mv.x & Displacement)
    strain = (mv.x & Displacement) * shear_first + (mv.y & Displacement) * shear_second
    bond_motion = strain(Displacement, directions)
    shear_response = bond_motion.adjugate()(response(bond_motion)).sum(axis=0)
    along = Displacement.dual_inverse()
    expected = -1.5 * stiffness * along + 1.5 * activity * (along | mv.xy)
    bond_cycle = bond_motion(cycle[:, None])
    bond_work = (response((bond_cycle[1:] + bond_cycle[:-1]) / 2) & (bond_cycle[1:] - bond_cycle[:-1])).sum(axis=1).sum(axis=0)
    area = (cycle[:-1] ^ cycle[1:]).sum(axis=0) / 2
    expected_work = -(3 * activity * mv.xy).scalar_product(area)

    # checks
    np.testing.assert_allclose(shear_response.kernel, expected.kernel, atol=1e-12)
    np.testing.assert_allclose(bond_work.kernel, loop_work(shear_response, cycle).kernel, atol=1e-12)
    np.testing.assert_allclose(bond_work.kernel, expected_work.kernel, atol=1e-12)


@pytest.mark.parametrize("stiffness, activity", [(0.9, 0), (0.9, 1.3), (0, 1.3)])
def test_lattice_motion_is_covariant_and_has_the_expected_strain_dissipation(stiffness: float, activity: float):
    n_columns, n_rows = 5, 4
    n_sites = n_columns * n_rows
    spacing = 0.8
    mesh, positions = core.triangular_patch(n_columns, n_rows, spacing)
    incidence = mesh.d[0]
    bonds = incidence * positions
    directions = bonds.normalized()
    gradient = incidence * Displacement
    drags = mv.scalar([[1]] * n_sites).field()
    generator = core.evolution(incidence, core.bond_response(directions, mv.xy, stiffness, activity), drags)
    displacement = mv.vector(np.random.default_rng(7).normal(scale=0.1, size=(n_sites, 2))).dual().field()
    velocity = generator(displacement)
    translation = 0 * positions + 0.4 * x - 0.2 * y
    rotation = positions | mv.xy

    # The gradient of half the sum of squared bond extensions reads the extensions
    # back as longitudinal nodal loads. Sideways activity does no work against it.
    along = directions.dual_inverse()
    strain_map = gradient.adjugate()(spdiag(along * (along & Displacement))(gradient))
    strain_gradient = strain_map(displacement)
    strain_rate = strain_gradient & velocity
    passive_decay = -stiffness * strain_gradient.scalar_norm_squared()

    mirror = (mv.x + 0.3 * mv.y).normalized()
    reflected_response = core.bond_response(mirror >> directions, mirror >> mv.xy, stiffness, activity)
    reflected_velocity = core.evolution(incidence, reflected_response, drags)(mirror >> displacement)

    # checks
    np.testing.assert_allclose(bonds.scalar_norm_squared().kernel, spacing**2, atol=1e-12)
    np.testing.assert_allclose(velocity.kernel.sum(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(generator(translation).kernel, 0, atol=1e-12)
    np.testing.assert_allclose(generator(rotation).kernel, 0, atol=1e-12)
    np.testing.assert_allclose(strain_rate.kernel.sum(), passive_decay.kernel.sum(), atol=1e-12)
    np.testing.assert_allclose(reflected_velocity.kernel, (mirror >> velocity).kernel, atol=1e-12)


def test_purely_active_evolution_preserves_the_strain_quadratic():
    n_columns, n_rows = 5, 4
    n_sites = n_columns * n_rows
    n_steps, dt = 25, 0.04
    mesh, positions = core.triangular_patch(n_columns, n_rows, 1)
    incidence = mesh.d[0]
    directions = (incidence * positions).normalized()
    drags = mv.scalar([[1]] * n_sites).field()
    generator = core.evolution(incidence, core.bond_response(directions, mv.xy, 0, 1.2), drags)
    initial = mv.vector(np.random.default_rng(5).normal(scale=0.1, size=(n_sites, 2))).dual().field()
    displacement = initial
    for _ in range(n_steps):
        displacement = core.rk4_step(generator, displacement, dt)
    along = directions.dual_inverse()
    initial_extensions = along & (incidence * initial)
    final_extensions = along & (incidence * displacement)

    # checks
    initial_quadratic = initial_extensions.scalar_norm_squared().kernel.sum()
    final_quadratic = final_extensions.scalar_norm_squared().kernel.sum()
    np.testing.assert_allclose(final_quadratic, initial_quadratic, rtol=2e-3)
    assert np.linalg.norm(displacement.kernel - initial.kernel) > 0.2
    np.testing.assert_allclose(displacement.kernel.sum(axis=0), initial.kernel.sum(axis=0), atol=1e-12)
