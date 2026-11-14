"""Tests cyclic work, reflection covariance, overdamped dynamics, and rendering in odd elastic lattices."""

from __future__ import annotations

import numpy as np
import pytest

from examples.mechanics.odd_elasticity import core, render, scenarios

mv, Displacement, Shear = core.mv, core.Displacement, core.Shear
x, y = core.x, core.y


@pytest.fixture
def cycle() -> core.Displacement:
    """An affine-stretched rotor circle forms a closed elliptical displacement polygon (`[n_samples + 1] Displacement-`)."""
    n_samples = 24
    angles = np.linspace(0, 2 * np.pi, n_samples + 1)
    turns = (-mv.xy * (angles / 2)).exp()                                    # [n_samples + 1] Rotor+
    circle = turns >> x                                                      # [n_samples + 1] Displacement-
    stretch = 0.18 * x * (mv.x & Displacement) + 0.11 * y * (mv.y & Displacement)  # Displacement- <- Displacement-
    return stretch(circle) + 0.07 * x - 0.03 * y                             # [n_samples + 1] Displacement-


def loop_work(response: core.Response, cycle: core.Displacement) -> core.Scalar:
    """Midpoint pairing integrates the mechanical work (`Scalar+`) of a linear `response` around a polygonal `cycle`."""
    return (response((cycle[1:] + cycle[:-1]) / 2) & (cycle[1:] - cycle[:-1])).sum(axis=0)


def test_cycle_work_is_oriented_area_and_changes_sign_on_reversal(cycle: core.Displacement) -> None:
    """Active bond work equals negative activity paired with swept area, whereas passive work vanishes."""
    activity = 0.8
    direction = (x + 0.3 * y).normalized()                                   # Displacement-
    response = core.bond_response(direction, mv.xy, 0.9, activity)           # Force+ <- Displacement-
    passive = core.bond_response(direction, mv.xy, 0.9, 0)                   # Force+ <- Displacement-
    work = loop_work(response, cycle)                                        # Scalar+
    area = (cycle[:-1] ^ cycle[1:]).sum(axis=0) / 2                          # Bivector+
    area_work = -activity * mv.xy.scalar_product(area)                       # Scalar+
    states = tuple(core.cycle_states(response, cycle))

    # checks
    assert ((work - area_work).abs() < 1e-12).all()
    assert (loop_work(passive, cycle).abs() < 1e-12).all()
    assert ((loop_work(response, cycle[::-1]) + work).abs() < 1e-12).all()
    assert ((states[-1][2] - area_work).abs() < 1e-12).all()
    assert (work > 0.04).all()


def test_mirror_transports_bond_force_and_preserves_cycle_work(cycle: core.Displacement) -> None:
    """Reflecting both bond direction and actuation plane transports forces covariantly and preserves work."""
    direction = (x + 0.4 * y).normalized()                                   # Displacement-
    mirror = (mv.x - 0.2 * mv.y).normalized()                                # Force+
    response = core.bond_response(direction, mv.xy, 0.9, 0.8)                # Force+ <- Displacement-
    reflected_response = core.bond_response(mirror >> direction, mirror >> mv.xy, 0.9, 0.8)  # Force+ <- Displacement-
    reflected_cycle = mirror >> cycle                                        # [...] Displacement-

    # checks
    assert ((reflected_response(reflected_cycle) - (mirror >> response(cycle))).norm() < 1e-12).all()
    assert ((loop_work(reflected_response, reflected_cycle) - loop_work(response, cycle)).abs() < 1e-12).all()


def test_three_bond_families_pull_back_to_an_odd_shear_response(cycle: core.Displacement) -> None:
    """Three sixty-degree bond families pull back to an isotropic odd shear modulus with triple cycle work."""
    stiffness, activity = 0.9, 0.8
    angles = np.arange(3) * np.pi / 3
    turns = (-mv.xy * (angles / 2)).exp()                                    # [3] Rotor+
    directions = turns >> x                                                  # [3] Displacement-
    response = core.bond_response(directions, mv.xy, stiffness, activity)    # [3] Force+ <- Displacement-
    bond_motion = core.shear_strain(Shear, directions)                       # [3] Displacement- <- Shear-
    shear_response = bond_motion.adjugate()(response(bond_motion)).sum(axis=0)  # Force+ <- Shear-
    duality = Displacement.dual_inverse()                                    # Force+ <- Displacement-
    expected = -1.5 * stiffness * duality + 1.5 * activity * (duality | mv.xy)  # Force+ <- Displacement-
    bond_cycle = bond_motion(cycle[:, None])                                 # [..., 3] Displacement-
    bond_work = (response((bond_cycle[1:] + bond_cycle[:-1]) / 2) & (bond_cycle[1:] - bond_cycle[:-1])).sum(axis=1).sum(axis=0)  # Scalar+
    area = (cycle[:-1] ^ cycle[1:]).sum(axis=0) / 2                          # Bivector+
    expected_work = -(3 * activity * mv.xy).scalar_product(area)             # Scalar+

    # checks
    assert (((shear_response - expected)(core.stack((x, y)))).norm() < 1e-12).all()
    assert ((bond_work - loop_work(shear_response, cycle)).abs() < 1e-12).all()
    assert ((bond_work - expected_work).abs() < 1e-12).all()


@pytest.mark.parametrize(("stiffness", "activity"), [(0.9, 0), (0.9, 1.3), (0, 1.3)])
def test_lattice_motion_is_covariant_and_has_the_expected_strain_dissipation(stiffness: float, activity: float) -> None:
    """Overdamped lattice velocities vanish on rigid motions, mirror covariantly, and dissipate only through stiffness."""
    n_columns, n_rows = 4, 3
    spacing = 0.8
    topology, coordinates = scenarios.triangular_patch(n_columns, n_rows, spacing)
    V, E, F = topology.n_elements
    positions = mv.vector(coordinates).dual().field()                        # Displacement-[V]
    incidence = topology.d[0]
    bonds = incidence * positions                                            # Displacement-[E]
    directions = bonds.normalized()                                          # Displacement-[E]
    drags = mv.scalar([[1]] * V).field()                                     # Scalar+[V]
    velocities = core.velocity_map(incidence, core.bond_response(directions, mv.xy, stiffness, activity), drags)
    displacement = mv.vector(np.random.default_rng(7).normal(scale=0.1, size=(V, 2))).dual().field()  # Displacement-[V]
    velocity = velocities(displacement)                                      # Displacement-[V]
    translation = 0 * positions + 0.4 * x - 0.2 * y                          # Displacement-[V]
    rotation = positions | mv.xy                                             # Displacement-[V]

    # Longitudinal strain gradient pairs orthogonally with transverse active forces.
    unit_forces = directions.dual_inverse()                                  # Force+[E]
    strain_map = core.lattice_response(incidence, unit_forces * (unit_forces & Displacement))
    strain_gradient = strain_map(displacement)                               # Force+[V]
    strain_rate = (strain_gradient & velocity).batch().sum()                 # Scalar+
    passive_decay = -stiffness * strain_gradient.scalar_norm_squared().batch().sum()  # Scalar+

    mirror = (mv.x + 0.3 * mv.y).normalized()                                # Force+
    reflected_response = core.bond_response(mirror >> directions, mirror >> mv.xy, stiffness, activity)
    reflected_velocity = core.velocity_map(incidence, reflected_response, drags)(mirror >> displacement)

    # checks
    assert ((bonds.scalar_norm_squared() - spacing**2).abs() < 1e-12).all()
    assert (velocity.batch().sum(axis=0).norm() < 1e-12).all()
    assert (velocities(translation).norm() < 1e-12).all()
    assert (velocities(rotation).norm() < 1e-12).all()
    assert ((strain_rate - passive_decay).abs() < 1e-12).all()
    assert ((reflected_velocity - (mirror >> velocity)).norm() < 1e-12).all()


def test_purely_active_evolution_preserves_the_strain_quadratic() -> None:
    """Zero-stiffness active dynamics conserves the sum of squared bond extensions under RK4 integration."""
    n_columns, n_rows = 4, 3
    n_steps, dt = 8, 0.04
    topology, coordinates = scenarios.triangular_patch(n_columns, n_rows, 1)
    V, E, F = topology.n_elements
    positions = mv.vector(coordinates).dual().field()                        # Displacement-[V]
    incidence = topology.d[0]
    directions = (incidence * positions).normalized()                        # Displacement-[E]
    drags = mv.scalar([[1]] * V).field()                                     # Scalar+[V]
    velocities = core.velocity_map(incidence, core.bond_response(directions, mv.xy, 0, 1.2), drags)
    initial = mv.vector(np.random.default_rng(5).normal(scale=0.1, size=(V, 2))).dual().field()  # Displacement-[V]
    displacement = initial                                                   # Displacement-[V]
    for _ in range(n_steps):
        displacement = core.rk4_step(velocities, displacement, dt)           # Displacement-[V]
    unit_forces = directions.dual_inverse()                                  # Force+[E]
    initial_quadratic = (unit_forces & (incidence * initial)).scalar_norm_squared().batch().sum()  # Scalar+
    final_quadratic = (unit_forces & (incidence * displacement)).scalar_norm_squared().batch().sum()  # Scalar+

    # checks
    assert ((final_quadratic - initial_quadratic).abs() < 1e-3 * initial_quadratic).all()
    assert ((displacement - initial).batch().norm().sum() > 0.1).all()
    assert (((displacement - initial).batch().sum(axis=0)).norm() < 1e-12).all()


def test_render_frame_generators_yield_rgb_frames() -> None:
    """Each frame generator yields RGB `uint8` frames on minimal single-step inputs without writing files."""
    anchor, reference, loop, bond_stream = scenarios.bond_cycle(2)
    bond_frame = next(render.bond_frames(anchor, reference, loop, 0.55, bond_stream))

    topology, coordinates = scenarios.triangular_patch(3, 3, 1)
    V, E, F = topology.n_elements
    patch_reference = mv.vector(coordinates).dual().field()                  # Displacement-[V]
    zero_force = (0 * mv.x + 0 * patch_reference.dual_inverse())             # Force+[V]
    patch_states = [(patch_reference, zero_force, mv.scalar([0])), (patch_reference, zero_force, mv.scalar([0]))]
    patch_frame = next(render.patch_frames(patch_reference, topology, 1.0, patch_states))

    directions = (topology.d[0] * patch_reference).normalized()              # Displacement-[E]
    drags = mv.scalar([[1]] * V).field()                                     # Scalar+[V]
    velocities = core.velocity_map(topology.d[0], core.bond_response(directions, mv.xy, 0.2, 1.0), drags)
    wave_stream = core.wave_states(patch_reference, topology.d[0], directions, velocities, 0 * patch_reference, 1, 1, 0.02)
    wave_frame = next(render.wave_frames(patch_reference, topology, 4.0, 0.04, 4, wave_stream))

    # checks
    for frame in (bond_frame, patch_frame, wave_frame):
        assert frame.ndim == 3 and frame.shape[2] == 3 and frame.dtype == np.uint8
