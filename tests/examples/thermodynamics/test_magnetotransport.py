"""Tests carrier Hall angles, compensated charge and entropy fluxes, Lorentz boost invariance, Onsager-Casimir reciprocity, and rendering for two-carrier magnetotransport."""

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.thermodynamics.magnetotransport import core, render, scenarios

mv = core.mv
carriers = scenarios.carriers
compensated_densities = np.full(2, 0.5)                                       # [2]
n_strengths = 5
strengths = np.linspace(0.0, scenarios.strength, n_strengths)                 # [n_strengths]


def field_sweep_states(topology: core.Topology, vertices: core.Point, high_contacts: np.ndarray,
                       low_contacts: np.ndarray, span: float, densities: np.ndarray) -> list:
    """Solves steady transport states across `n_strengths` magnetic field values at fixed carrier `densities` (`[2]`)."""
    return list(scenarios.conductor_states(
        topology, vertices, high_contacts, low_contacts, span, strengths * mv.xy, carriers,
        np.broadcast_to(densities, (n_strengths, 2))))


def test_each_kind_turns_by_its_own_rotor():
    magnetic = scenarios.strength * mv.xy                                     # [] Magnetic+
    hall_rotors = 1 - scenarios.valences * scenarios.mobilities * magnetic    # [2] Rotor+
    angles = hall_rotors.normalized().log().scalar_product(~mv.xy)            # [2] Scalar+
    expected_angles = -scenarios.valences * np.arctan(scenarios.mobilities * scenarios.strength)
    assert ((angles - expected_angles).abs() < 1e-12).all()

    flux = core.response(magnetic, *carriers, compensated_densities, scenarios.conductivity)(mv.xt)   # [] Flux-
    charge_along, charge_across = mv.xt & flux, mv.yt & flux                  # [] Scalar+, [] Scalar+
    entropy_along, entropy_across = mv.xs & flux, mv.ys & flux                # [] Scalar+, [] Scalar+
    assert charge_across.abs() < 1e-12 * charge_along.abs()
    assert entropy_along.abs() < 1e-12 * entropy_across.abs()
    assert entropy_across.abs() > charge_along.abs()


def compensated_bar() -> tuple:
    """Solves a coarse current-driven Hall bar at carrier compensation and returns its simplex gradients, incidence, magnetic field, transport map, vertex potentials, and triangle drives."""
    topology, vertices, left_contacts, right_contacts, _, _ = scenarios.bar(n_rows=4, n_columns=12)
    areas, gradients = core.triangles(topology, vertices)
    corners = topology.incidence[2, 0]                                        # [3] Scalar+[F <- V]
    magnetic = scenarios.strength * mv.xy                                     # [] Magnetic+
    transport_map = core.response(magnetic, *carriers, compensated_densities, scenarios.conductivity)
    electric_potentials, _ = core.steady_potentials(
        topology, areas, gradients, transport_map, left_contacts, right_contacts, scenarios.current,
        scenarios.temperature_fall * scenarios.length)
    triangle_drive = -(gradients ^ (corners * electric_potentials)).sum(axis=0)   # Drive+[F]
    return gradients, corners, magnetic, transport_map, electric_potentials, triangle_drive, core.drifts(magnetic, *carriers)(triangle_drive)


def test_moving_observer_reads_another_electric_field_and_the_same_dissipation():
    gradients, corners, magnetic, transport_map, electric_potentials, triangle_drive, _ = compensated_bar()
    electric_field = -(gradients ^ ((corners * (electric_potentials & core.time)) * mv.t)).sum(axis=0)   # Bivector+[F]
    rapidity = 0.3
    boost = (mv.xt * (rapidity / 2)).exp()                                    # [] Rotor+
    spacetime_field = electric_field + magnetic                               # Bivector+[F]
    swept_across = core.y_direction & core.time                               # [] Flux-
    rest_reading = spacetime_field & swept_across                             # Scalar+[F]
    moving_reading = (boost >> spacetime_field) & swept_across                # Scalar+[F]
    lorentz_reading = np.cosh(rapidity) * rest_reading + np.sinh(rapidity) * scenarios.strength
    assert ((moving_reading - lorentz_reading).abs() < 1e-10).all()
    flux = transport_map(triangle_drive)                                      # Flux-[F]
    assert ((((boost >> triangle_drive) & (boost >> flux)) - (triangle_drive & flux)).abs() < 1e-10).all()


def test_hall_bar_at_compensation():
    topology, vertices, left_contacts, right_contacts, plus_probes, minus_probes = scenarios.bar(n_rows=4, n_columns=12)
    states = field_sweep_states(topology, vertices, left_contacts, right_contacts, scenarios.length,
                                compensated_densities)
    resistance, hall, *_ = scenarios.bar_readings(topology, plus_probes, minus_probes, states)
    assert (hall.abs() < 1e-12 * resistance.max()).all()
    assert ((resistance[1:] - resistance[:-1]) > 0).all() and resistance[-1] > 2 * resistance[0]


def test_response_reversed_in_the_field_is_its_adjugate():
    magnetic = np.array([-1, 0, 1]) * scenarios.strength * mv.xy              # [3] Magnetic+
    densities = np.array([0.3, 0.7])                                          # [2]
    reversed_field, zero_field, forward_field = core.response(magnetic, *carriers, densities, scenarios.conductivity)
    scale = np.abs(forward_field.kernel).max()
    np.testing.assert_allclose(forward_field.kernel, reversed_field.adjugate().kernel, atol=1e-12 * scale)
    np.testing.assert_allclose(zero_field.kernel, zero_field.adjugate().kernel, atol=1e-12 * scale)

    n_drives = 64
    sample_drives = mv(core.Drive, np.random.default_rng(0).normal(size=(n_drives, 4)))   # [n_drives] Drive+
    dissipation = sample_drives & stack((reversed_field, zero_field, forward_field))[:, None](sample_drives)   # [3, n_drives] Scalar+
    assert (dissipation > 0).all()
    antisymmetric_part = (forward_field - reversed_field) / 2                 # [] Flux- <- Drive+
    assert (sample_drives & antisymmetric_part(sample_drives)).abs().max() < 1e-12 * dissipation.max()


def test_scenes_run_and_draw():
    bar_topology, bar_vertices, left_contacts, right_contacts, _, _ = scenarios.bar(2, 6)
    ring_topology, ring_vertices, inner_contacts, outer_contacts = scenarios.ring(2, 8)
    field, densities = scenarios.magnetic[:1], scenarios.densities[:1]
    first_frame = next(zip(
        field,
        scenarios.conductor_states(bar_topology, bar_vertices, left_contacts, right_contacts, scenarios.length, field,
                                   carriers, densities),
        scenarios.conductor_states(ring_topology, ring_vertices, inner_contacts, outer_contacts,
                                   1 - scenarios.inner_radius, field, carriers, densities)))
    resolution = 20
    next(render.heat_driven_animation(bar_topology, bar_vertices, (left_contacts, right_contacts), ring_topology,
                                      ring_vertices, (inner_contacts, outer_contacts), resolution, resolution,
                                      first_frame, [first_frame]))
    render.turns_figure(mv.xt, scenarios.end_fluxes())
    render.conductors_figure(bar_topology, bar_vertices, (left_contacts, right_contacts), ring_topology, ring_vertices,
                             (inner_contacts, outer_contacts), resolution, resolution)
    plt.close("all")
