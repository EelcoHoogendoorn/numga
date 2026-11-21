"""Tests carrier Hall angles, compensated charge and entropy fluxes, Lorentz boost invariance, Onsager-Casimir reciprocity, and rendering for two-carrier magnetotransport."""

import matplotlib.pyplot as plt
import numpy as np

from numga import dim, stack
from examples.thermodynamics.magnetotransport import core, render, scenarios

mv = core.mv
# Unit speed of light enters only where a Lorentz boost mixes electric and magnetic fields.
c = mv.scalar(1.0, unit=dim.length / dim.time)                                # [] Scalar+, length/time
power_density_dim = dim.voltage * dim.current / dim.length**2
carriers = scenarios.carriers
compensated_densities = np.full(2, 0.5) * scenarios.total_density             # [2] Scalar+, current*time/length^2
n_strengths = 5
strengths = np.linspace(0.0, 1.0, n_strengths) * scenarios.strength           # [n_strengths] Scalar+, voltage*time/length^2


def field_sweep_states(topology: core.Topology, vertices: core.Point, high_contacts: np.ndarray,
                       low_contacts: np.ndarray, span: core.Scalar, densities: core.Scalar) -> list:
    """Solves steady transport states across `n_strengths` magnetic field values at fixed carrier `densities` (`[2]`)."""
    return list(scenarios.conductor_states(
        topology, vertices, high_contacts, low_contacts, span, strengths * mv.xy, carriers,
        densities[None].broadcast_to((n_strengths, 2))))


def test_each_kind_turns_by_its_own_rotor():
    magnetic = scenarios.strength * mv.xy                                     # [] Magnetic+, voltage*time/length^2
    hall_rotors = 1 - scenarios.valences * scenarios.mobilities * magnetic    # [2] Rotor+
    angles = hall_rotors.normalized().log().scalar_product(~mv.xy)            # [2] Scalar+
    expected_angles = -scenarios.valences * (scenarios.mobilities * scenarios.strength).arctan()
    assert ((angles - expected_angles).abs() < 1e-12).all()

    flux = core.response(magnetic, *carriers, compensated_densities, scenarios.conductivity)(
        mv.xt * (dim.voltage / dim.length))                                   # [] Flux-
    charge_along, charge_across = mv.xt & flux, mv.yt & flux                  # [] Scalar+, current/length
    entropy_along, entropy_across = mv.xs & flux, mv.ys & flux                # [] Scalar+, current*voltage/(temperature*length)
    assert charge_across.abs() < 1e-12 * charge_along.abs()
    assert entropy_along.abs() < 1e-12 * entropy_across.abs()
    assert entropy_across.abs() > (scenarios.entropies_per_charge[1] * charge_along).abs()


def compensated_bar() -> tuple:
    """Solves a coarse current-driven Hall bar at carrier compensation and returns its simplex gradients, incidence, magnetic field, transport map, vertex potentials, and triangle drives."""
    topology, vertices, left_contacts, right_contacts, _, _ = scenarios.bar(n_rows=4, n_columns=12)
    areas, gradients = core.triangles(topology, vertices)
    corners = topology.incidence[2, 0]                                        # [3] Scalar+[F <- V]
    magnetic = scenarios.strength * mv.xy                                     # [] Magnetic+, voltage*time/length^2
    transport_map = core.response(magnetic, *carriers, compensated_densities, scenarios.conductivity)
    electric_potentials, _ = core.steady_potentials(
        topology, areas, gradients, transport_map, left_contacts, right_contacts, scenarios.current,
        scenarios.temperature_fall * scenarios.length)
    triangle_drive = -(gradients ^ (corners * electric_potentials)).sum(axis=0)   # Drive+[F], voltage/length | temperature/length
    return gradients, corners, magnetic, transport_map, electric_potentials, triangle_drive, core.drifts(magnetic, *carriers)(triangle_drive)


def test_moving_observer_reads_another_electric_field_and_the_same_dissipation():
    gradients, corners, magnetic, transport_map, electric_potentials, triangle_drive, _ = compensated_bar()
    electric_field = -(gradients ^ ((corners * (electric_potentials & core.time)) * mv.t)).sum(axis=0)   # Bivector+[F], voltage/length
    rapidity = 0.3
    boost = (mv.xt * (rapidity / 2)).exp()                                    # [] Rotor+
    # Combining electric field with c * magnetic puts the relativistic electromagnetic bivector on one dimension.
    spacetime_field = electric_field + c * magnetic                           # Bivector+[F], voltage/length
    swept_across = core.y_direction & core.time                               # [] Flux-
    rest_reading = spacetime_field & swept_across                             # Scalar+[F], voltage/length
    moving_reading = (boost >> spacetime_field) & swept_across                # Scalar+[F], voltage/length
    lorentz_reading = np.cosh(rapidity) * rest_reading + np.sinh(rapidity) * c * scenarios.strength
    assert ((moving_reading - lorentz_reading).abs() < 1e-10 * c * scenarios.strength).all()
    flux = transport_map(triangle_drive)                                      # Flux-[F]
    assert ((((boost >> triangle_drive) & (boost >> flux)) - (triangle_drive & flux)).abs() < mv.scalar(1e-10, unit=power_density_dim)).all()


def test_hall_bar_at_compensation():
    topology, vertices, left_contacts, right_contacts, plus_probes, minus_probes = scenarios.bar(n_rows=4, n_columns=12)
    states = field_sweep_states(topology, vertices, left_contacts, right_contacts, scenarios.length,
                                compensated_densities)
    resistance, hall, *_ = scenarios.bar_readings(topology, plus_probes, minus_probes, states)
    assert (hall.abs() < 1e-12 * resistance.max()).all()
    assert ((resistance[1:] - resistance[:-1]) > 0).all() and resistance[-1] > 2 * resistance[0]


def test_response_reversed_in_the_field_is_its_adjugate():
    magnetic = np.array([-1, 0, 1]) * scenarios.strength * mv.xy              # [3] Magnetic+, voltage*time/length^2
    densities = np.array([0.3, 0.7]) * scenarios.total_density                # [2] Scalar+, current*time/length^2
    reversed_field, zero_field, forward_field = core.response(magnetic, *carriers, densities, scenarios.conductivity)
    scale = np.abs(forward_field.kernel).max()
    np.testing.assert_allclose(forward_field.kernel, reversed_field.adjugate().kernel, atol=1e-12 * scale)
    np.testing.assert_allclose(zero_field.kernel, zero_field.adjugate().kernel, atol=1e-12 * scale)

    n_drives = 64
    rng = np.random.default_rng(0)
    electric_samples = (mv(core.Drive.derive.touching(core.space, core.Algebra("t-")), rng.normal(size=(n_drives, 2)))
                        * (dim.voltage / dim.length))
    thermal_samples = (mv(core.Drive.derive.touching(core.space, core.Algebra("s0")), rng.normal(size=(n_drives, 2)))
                       * (dim.temperature / dim.length))
    sample_drives = electric_samples + thermal_samples                        # [n_drives] Drive+, voltage/length | temperature/length
    dissipation = sample_drives & stack((reversed_field, zero_field, forward_field))[:, None](sample_drives)   # [3, n_drives] Scalar+, current*voltage/length^2
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
                                   scenarios.outer_radius - scenarios.inner_radius, field, carriers, densities)))
    resolution = mv.scalar(20.0, unit=dim.pixel / dim.length)
    next(render.heat_driven_animation(bar_topology, bar_vertices, (left_contacts, right_contacts), ring_topology,
                                      ring_vertices, (inner_contacts, outer_contacts), resolution, resolution,
                                      first_frame, [first_frame]))
    render.turns_figure(mv.xt * (dim.voltage / dim.length), scenarios.end_fluxes())
    render.conductors_figure(bar_topology, bar_vertices, (left_contacts, right_contacts), ring_topology, ring_vertices,
                             (inner_contacts, outer_contacts), resolution, resolution)
    plt.close("all")


def test_readings_at_one_place():
    resistance, hall, seebeck, nernst, righi_leduc = scenarios.interior_readings()
    middle = scenarios.n_frames // 2
    hall_scale = scenarios.strength / scenarios.total_density
    assert ((hall[[0, -1]] - hall_scale / scenarios.valences).abs() < 1e-9 * scenarios.sheet_resistance).all()
    assert ((resistance[[0, -1]] - scenarios.sheet_resistance).abs() < 1e-9 * scenarios.sheet_resistance).all()
    assert hall[middle].abs() < 1e-9 * scenarios.sheet_resistance
    assert nernst.abs().argmax() == middle
    assert (righi_leduc[[0, middle, -1]].abs() < 1e-10).all()
