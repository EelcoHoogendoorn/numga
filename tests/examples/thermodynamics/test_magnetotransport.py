"""Galvanomagnetic and thermomagnetic effects of electrons and holes: each kind's Hall angle the angle of its own rotor,
opposite for the two, whose turns cancel in the charge a compensated conductor carries and add in its entropy; each
kind's drift balancing one commutator of the field and the drive with its moving parcel, of which the magnetic field
spends no power; with one kind of carrier, a bar's Hall voltage linear in the field and its resistance along it
unchanged; at compensation, no Hall voltage and a resistance rising with the field; the Corbino ring's resistance
rising as one plus the square of the mobility times the field at every balance; the Nernst signal largest at
compensation and nothing with one kind, and the Righi-Leduc signal only with both kinds, unbalanced; the response
reversed in the field its adjugate, its dissipation never negative and the same for a moving observer, who reads
another electric field; and the scenes drawing."""

import numpy as np

from numga import stack
from examples.thermodynamics.magnetotransport import core, render, scenarios

mv = core.mv
# The carriers' constants, as the core's functions take them.
carriers = (scenarios.valences, scenarios.entropies_per_charge, scenarios.mobilities)
# The densities of the carriers with electrons only, at compensation and with holes only; and the field ramped up from
# nothing to the scenario's strength.
electrons, compensated, holes = np.eye(2)[0], np.full(2, 0.5), np.eye(2)[1]  # [kinds] each
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


def test_the_drift_balances_one_commutator_and_the_field_does_no_work():
    gradients, corners, magnetic, response, electric, drive, velocities = compensated_bar()
    parcels = scenarios.valences * (core.time + scenarios.entropies_per_charge * core.entropy)   # [kinds] Parcel-
    moving_parcels = parcels + scenarios.valences * velocities               # [kinds] Point-[faces]
    pushes = (magnetic + drive).commutator(moving_parcels)                    # [kinds] Point-[faces]
    # Read through space, the whole push is the drag, the velocity over the mobility: the rotor's inverse solves the
    # balance written as one commutator.
    through_space = stack((mv.x, mv.y))[:, None]
    np.testing.assert_allclose((through_space & pushes).to_array(),
                               (through_space & (velocities / scenarios.mobilities)).to_array(), atol=1e-10)
    # Along time, the power spent on the carriers: the magnetic field spends none, and summed over the kinds at their
    # densities it is the electric field's reading of the charge flux.
    assert np.abs((mv.t & magnetic.commutator(moving_parcels)).to_array()).max() < 1e-12
    electric_field = -(gradients ^ ((corners * (electric & core.time)) * mv.t)).sum(axis=0)   # Bivector+[faces]
    power = (compensated * (mv.t & pushes)).sum(axis=0)                       # Scalar+[faces]
    np.testing.assert_allclose(power.to_array(), (electric_field & response(drive)).to_array(), atol=1e-10)


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


def test_hall_bar_with_one_kind_of_carrier():
    topology, vertices, left, right, plus, minus = scenarios.bar()
    for densities in (electrons, holes):
        states = ramp(topology, vertices, left, right, scenarios.length, densities)
        resistance, hall, *_ = (reading.to_array() for reading in scenarios.bar_readings(topology, plus, minus,
                                                                                         states))
        # The Hall resistance is linear in the field, its slope one over the carriers' density of charge, less the
        # share the contacts short at the bar's middle; the resistance along the bar stays as it was.
        slopes = hall[1:] / strengths[1:]
        np.testing.assert_allclose(slopes, slopes[0], rtol=1e-2)
        np.testing.assert_allclose(slopes, 1 / (densities @ scenarios.valences), rtol=3e-2)
        np.testing.assert_allclose(resistance, resistance[0], rtol=1e-2)


def test_hall_bar_at_compensation():
    topology, vertices, left, right, plus, minus = scenarios.bar()
    states = ramp(topology, vertices, left, right, scenarios.length, compensated)
    resistance, hall, *_ = (reading.to_array() for reading in scenarios.bar_readings(topology, plus, minus, states))
    # No Hall voltage at any field, and the resistance along the bar rising with it.
    np.testing.assert_allclose(hall, 0.0, atol=1e-12 * resistance.max())
    assert np.all(np.diff(resistance) > 0) and resistance[-1] > 2 * resistance[0]


def test_corbino_resistance_rises_at_every_balance():
    topology, vertices, inner, outer = scenarios.ring()
    for densities in (electrons, compensated, holes):
        states = ramp(topology, vertices, inner, outer, 1 - scenarios.inner_radius, densities)
        resistances = scenarios.ring_resistances(topology, inner, outer, states).to_array()
        np.testing.assert_allclose(resistances / resistances[0], 1 + (scenarios.mobilities[0] * strengths) ** 2,
                                   rtol=1e-10)


def test_heat_driven_bar_across_the_balance():
    topology, vertices, left, right, plus, minus = scenarios.bar()
    states = list(scenarios.states(topology, vertices, left, right, scenarios.length, scenarios.magnetic,
                                   scenarios.densities))
    readings = scenarios.bar_readings(topology, plus, minus, states)
    _, hall, seebeck, nernst, righi_leduc = (reading.to_array() for reading in readings)
    middle = scenarios.n_frames // 2
    # The Hall voltage vanishes at compensation, where the Nernst signal is largest; with one kind of carrier there is
    # no Nernst signal, and the Seebeck signal is that kind's entropy per charge.
    assert abs(hall[middle]) < 1e-12 * np.abs(hall).max()
    assert np.abs(nernst).argmax() == middle
    np.testing.assert_allclose(nernst[[0, -1]], 0.0, atol=1e-10 * np.abs(nernst).max())
    np.testing.assert_allclose(seebeck[[0, -1]], scenarios.entropies_per_charge, rtol=1e-10)
    # The Righi-Leduc signal, a fall of temperature across the bar, needs both kinds, unbalanced: nothing with one kind,
    # whose carriers stand still, and nothing at compensation, where both drift straight down the fall; between, it
    # is plain, the lower side the warmer where electrons are the more numerous and the upper where holes are.
    np.testing.assert_allclose(righi_leduc[[0, middle, -1]], 0.0, atol=1e-10)
    electron_side, hole_side = righi_leduc[1:middle], righi_leduc[middle + 1:-1]
    assert electron_side.min() > 0 > hole_side.max() and np.abs(righi_leduc).max() > 0.1
    render.draw_readings(scenarios.balances, *readings)


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


def test_scenes_draw():
    render.draw_turns(mv.xt, scenarios.end_fluxes())
    bar_topology, bar_vertices, left, right, _, _ = scenarios.bar()
    ring_topology, ring_vertices, inner, outer = scenarios.ring()
    render.draw_conductors(bar_topology, bar_vertices, (left, right), ring_topology, ring_vertices, (inner, outer),
                           scenarios.bar_resolution, scenarios.ring_resolution)
    conductors = (bar_topology, bar_vertices, (left, right), ring_topology, ring_vertices, (inner, outer),
                  scenarios.bar_resolution, scenarios.ring_resolution)
    # The first frame of the sweep, and the conductors in no field, where the field's patch is a point.
    first, unturned = (next(zip(field,
                                scenarios.states(bar_topology, bar_vertices, left, right, scenarios.length, field,
                                                 scenarios.densities[:1]),
                                scenarios.states(ring_topology, ring_vertices, inner, outer,
                                                 1 - scenarios.inner_radius, field, scenarios.densities[:1])))
                       for field in (scenarios.magnetic[:1], 0 * scenarios.magnetic[:1]))
    for animate, frames in ((render.animate_current_driven, [first]), (render.animate_heat_driven, [first, unturned])):
        for frame in animate(*conductors, first, frames):
            assert frame.shape[-1] == 3
