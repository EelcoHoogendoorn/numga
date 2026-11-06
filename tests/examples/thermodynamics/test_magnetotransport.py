"""Galvanomagnetic and thermomagnetic effects of electrons and holes: each kind's Hall angle the angle of its own rotor,
opposite for the two, whose turns cancel in the charge a compensated conductor carries and add in its entropy; at
compensation, no Hall voltage and a resistance rising with the field; the response reversed in the field its adjugate,
its dissipation never negative and the same for a moving observer, who reads another electric field; and a coarse bar
and ring drawn."""

import numpy as np

from numga import stack
from examples.thermodynamics.magnetotransport import core, render, scenarios

mv = core.mv
# The carriers' constants, as the core's functions take them.
carriers = (scenarios.valences, scenarios.entropies_per_charge, scenarios.mobilities)
# The densities of the carriers at compensation; and the field ramped up from nothing to the scenario's strength.
compensated = np.full(2, 0.5)                                                # [kinds]
strengths = np.linspace(0.0, scenarios.strength, 5)                          # [strengths]


def ramp(topology: core.Topology, vertices: core.Point, high: np.ndarray, low: np.ndarray, span: float,
         densities: np.ndarray) -> list:
    """A conductor's steady states as the field is ramped up, at these densities of the carriers `[kinds]`."""
    return list(scenarios.states(topology, vertices, high, low, span, strengths * mv.xy,
                                 np.broadcast_to(densities, (len(strengths), 2))))


def test_each_kind_turns_by_its_own_rotor():
    magnetic = scenarios.strength * mv.xy                                     # [] Bivector+
    hall = 1 - scenarios.valences * scenarios.mobilities * magnetic          # [kinds] Even+
    # Each kind's Hall angle, the angle of its rotor, is the angle whose tangent is the mobility times the field, the
    # electrons' and the holes' opposite.
    angles = hall.normalized().log().scalar_product(~mv.xy).to_array()      # [kinds]
    np.testing.assert_allclose(angles, -scenarios.valences * np.arctan(scenarios.mobilities * scenarios.strength),
                               atol=1e-12)
    # A unit field along the bar drives, at compensation, a flux whose charge runs along the field and whose entropy
    # runs across it: the turns cancel in the one reading and add in the other.
    flux = core.response(magnetic, *carriers, compensated, scenarios.conductivity)(mv.xt)   # [] Flux-
    charge_along, charge_across = (mv.xt & flux).to_array(), (mv.yt & flux).to_array()
    entropy_along, entropy_across = (mv.xs & flux).to_array(), (mv.ys & flux).to_array()
    assert abs(charge_across) < 1e-12 * abs(charge_along)
    assert abs(entropy_along) < 1e-12 * abs(entropy_across)
    assert abs(entropy_across) > abs(charge_along)


def compensated_bar() -> tuple:
    """The current-driven bar at compensation in the scenario's field: its hat gradients and corners, the field, the
    response, the potentials, each triangle's drive and each kind's velocity there."""
    topology, vertices, left, right, _, _ = scenarios.bar()
    areas, gradients = core.triangles(topology, vertices)
    corners = topology.incidence[2, 0]                                        # [corners] Scalar+[faces <- vertices]
    magnetic = scenarios.strength * mv.xy                                     # [] Bivector+
    response = core.response(magnetic, *carriers, compensated, scenarios.conductivity)
    electric, _ = core.steady(topology, areas, gradients, response, left, right, scenarios.current,
                              scenarios.temperature_fall * scenarios.length)
    drive = -(gradients ^ (corners * electric)).sum(axis=0)                  # Drive+[faces]
    return gradients, corners, magnetic, response, electric, drive, core.drifts(magnetic, *carriers)(drive)


def test_moving_observer_reads_another_electric_field_and_the_same_dissipation():
    gradients, corners, magnetic, response, electric, drive, _ = compensated_bar()
    electric_field = -(gradients ^ ((corners * (electric & core.time)) * mv.t)).sum(axis=0)   # Bivector+[faces]
    rapidity = 0.3
    boost = (mv.xt * (rapidity / 2)).exp()                                    # [] Rotor+
    # The electric field across the bar that an observer moving along it reads has the magnetic field mixed in, as
    # Lorentz has it; the dissipation, the drive's reading of the flux it drives, is the same.
    field = electric_field + magnetic                                         # Bivector+[faces]
    swept_across = core.across & core.time                                    # [] Flux-
    rest_reading = (field & swept_across).to_array()
    moving_reading = ((boost >> field) & swept_across).to_array()
    lorentz = np.cosh(rapidity) * rest_reading + np.sinh(rapidity) * scenarios.strength
    np.testing.assert_allclose(moving_reading, lorentz, atol=1e-10)
    flux = response(drive)                                                    # Flux-[faces]
    np.testing.assert_allclose(((boost >> drive) & (boost >> flux)).to_array(), (drive & flux).to_array(), atol=1e-10)


def test_hall_bar_at_compensation():
    topology, vertices, left, right, plus, minus = scenarios.bar()
    states = ramp(topology, vertices, left, right, scenarios.length, compensated)
    resistance, hall, *_ = (reading.to_array() for reading in scenarios.bar_readings(topology, plus, minus, states))
    # No Hall voltage at any field, and the resistance along the bar rising with it.
    np.testing.assert_allclose(hall, 0.0, atol=1e-12 * resistance.max())
    assert np.all(np.diff(resistance) > 0) and resistance[-1] > 2 * resistance[0]


def test_response_reversed_in_the_field_is_its_adjugate():
    magnetic = np.array([-1, 0, 1]) * scenarios.strength * mv.xy              # [3] Bivector+
    # A balance off compensation, so that each kind's part and their difference all show.
    densities = np.array([0.3, 0.7])                                          # [kinds]
    reversed_field, none, field = core.response(magnetic, *carriers, densities, scenarios.conductivity)
    scale = np.abs(field.kernel).max()
    # Onsager-Casimir: the cross coefficients between heat and charge are antisymmetric under reversing the field, the
    # response's adjugate; with no field, Onsager's symmetry.
    np.testing.assert_allclose(field.kernel, reversed_field.adjugate().kernel, atol=1e-12 * scale)
    np.testing.assert_allclose(none.kernel, none.adjugate().kernel, atol=1e-12 * scale)
    # The dissipation, a drive's reading of the flux it drives, is never negative, and the antisymmetric part of the
    # response, odd in the field, produces nothing.
    drives = mv(core.Drive, np.random.default_rng(0).normal(size=(500, 4)))   # [drives] Drive+
    dissipation = (drives & stack((reversed_field, none, field))[:, None](drives)).to_array()   # [3, drives]
    assert dissipation.min() > 0
    antisymmetric = (field - reversed_field) / 2
    assert np.abs((drives & antisymmetric(drives)).to_array()).max() < 1e-12 * dissipation.max()


def test_scenes_run_and_draw():
    # A coarse bar and ring, their steady states at the first frame's field and balance, drawn small, driven by heat.
    bar_topology, bar_vertices, left, right, _, _ = scenarios.bar(n_rows=2, n_columns=6)
    ring_topology, ring_vertices, inner, outer = scenarios.ring(n_rings=2, n_sectors=8)
    field, densities = scenarios.magnetic[:1], scenarios.densities[:1]
    first = next(zip(field,
                     scenarios.states(bar_topology, bar_vertices, left, right, scenarios.length, field, densities),
                     scenarios.states(ring_topology, ring_vertices, inner, outer, 1 - scenarios.inner_radius, field,
                                      densities)))
    resolution = 20
    next(render.animate_heat_driven(bar_topology, bar_vertices, (left, right), ring_topology, ring_vertices,
                                    (inner, outer), resolution, resolution, first, [first]))
