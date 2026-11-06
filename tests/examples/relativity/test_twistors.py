"""Twistor incidence, the straight rays of the Robinson congruence, and the fibres and field lines along them."""

import numpy as np

from examples.relativity.twistors import core
from examples.util.topology import Topology


TOLERANCE = 1e-11
SAMPLES = 24


def test_robinson_twistors_stay_orthogonal_along_straight_future_null_rays():
    rng = np.random.default_rng(823)
    events = core.mv(core.Event.derive.dual, rng.normal(size=(SAMPLES, 4))).dual()
    twistors = core.selection(core.point(events), core.robinson_twistor)
    extrinsic_volume = (8 * core.Full).cast(core.ExtrinsicTwistor)(
        core.Full * (core.ExtrinsicTwistor * core.projector))(core.exact.xyztuv)
    directions = core.robinson(events)
    times = np.linspace(-1.0, 1.0, 9)
    along_rays = events[:, None] + directions[:, None] * times

    # The event's selected twistor pairs to zero with the fixed twistor and with
    # its turn by the volume. Every event along its straight light ray selects the
    # same propagation direction and remains incident with the initial twistor.
    np.testing.assert_allclose(core.pairing(twistors, core.robinson_twistor).kernel,
                               0, atol=TOLERANCE)
    np.testing.assert_allclose(core.pairing(twistors, extrinsic_volume(core.robinson_twistor)).kernel,
                               0, atol=TOLERANCE)
    np.testing.assert_allclose(directions.squared().kernel, 0, atol=TOLERANCE)
    np.testing.assert_allclose(((core.mv.t & directions) - 1).kernel, 0, atol=TOLERANCE)
    np.testing.assert_allclose((core.robinson(along_rays) - directions[:, None]).kernel,
                               0, atol=TOLERANCE)
    np.testing.assert_allclose(core.incidence(core.point(along_rays), twistors[:, None]).kernel,
                               0, atol=TOLERANCE)


POLARS = np.pi * np.array([0.4, 0.65])
PER_CIRCLE = 3
FIBRE_SAMPLES = 192
TIMES = np.array([-0.8, 0.0, 0.8])
# Centred differences of sampled curves resolve their tangents to about this accuracy.
TANGENT_TOLERANCE = 0.003


def test_flow_lines_follow_the_robinson_directions():
    curves = core.fibres(POLARS, PER_CIRCLE, FIBRE_SAMPLES)
    steps = curves[..., 2:] - curves[..., :-2]
    tangent = steps / steps.squared().square_root()
    flow = core.robinson(curves[..., 1:-1]).cast(core.Spatial)

    # checks: the sampled circles run along the spatial directions of the congruence.
    np.testing.assert_allclose((tangent & flow).kernel, 0, atol=TANGENT_TOLERANCE)


def test_field_lines_follow_their_fields_along_the_energy_flow_rays():
    from examples.relativity.twistors import scenarios

    curves = core.fibres(POLARS, PER_CIRCLE, FIBRE_SAMPLES)
    electric = scenarios.electric_turn >> curves
    magnetic = (core.mv.yz * (np.pi / 4)).exp() >> curves
    times = TIMES[:, None, None, None]
    electric_events = electric + core.robinson(electric) * times
    magnetic_events = magnetic + core.robinson(magnetic) * times
    # The fields' parts through time; the magnetic part turned into time by the spacetime volume.
    temporal = core.Bivector.derive.containing("t")
    electric_field = core.hopfion(electric_events).cast(temporal)[..., 1:-1]
    magnetic_field = (core.spacetime_volume * core.hopfion(magnetic_events)).cast(temporal)[..., 1:-1]
    electric_steps = electric_events[..., 2:] - electric_events[..., :-2]
    magnetic_steps = magnetic_events[..., 2:] - magnetic_events[..., :-2]
    # The planes the lines' unit steps sweep through time.
    electric_sweeps = ((electric_steps / electric_steps.squared().square_root()) & core.time).dual()
    magnetic_sweeps = ((magnetic_steps / magnetic_steps.squared().square_root()) & core.time).dual()

    # checks: after their material points have followed the straight energy-flow rays, the electric
    # and the magnetic lines still run along their fields.
    np.testing.assert_allclose(
        electric_sweeps.commutator(electric_field / electric_field.squared().square_root()).kernel, 0, atol=TANGENT_TOLERANCE,
    )
    np.testing.assert_allclose(
        magnetic_sweeps.commutator(magnetic_field / magnetic_field.squared().square_root()).kernel, 0, atol=TANGENT_TOLERANCE,
    )


def test_two_fibres_link_once():
    curves = core.fibres(POLARS, PER_CIRCLE, FIBRE_SAMPLES)
    # Two closed fibres: their steps, from each sample to the next and the last back to the first, and the steps'
    # midpoints.
    loop = Topology.loops([FIBRE_SAMPLES])
    fibres = curves[0, :2].field()
    first_steps, second_steps = (loop.d[0] * fibres).batch()
    first_midpoints, second_midpoints = (loop.incidence[1, 0].mean(axis=0) * fibres).batch()
    offsets = first_midpoints[:, None] - second_midpoints[None, :]
    distance_squared = offsets.scalar_norm_squared()
    # The volume the three directions span, read by the volume of space.
    volume = core.mv.xyz & (offsets & first_steps[:, None] & second_steps[None, :])
    linking = (volume
               / (distance_squared * distance_squared.square_root())).sum() / (4 * np.pi)

    # checks: the Gauss integral counts one linking of two distinct closed fibres.
    np.testing.assert_allclose(np.abs(linking.kernel), 1, rtol=0, atol=0.0001)


def test_incidence_rays_meet_their_events_at_rest_and_boosted():
    from examples.relativity.twistors import scenarios

    twistors, events, rays = scenarios.incidence()
    rays_of_twistors = core.ray_readout(twistors, twistors)
    recovered = core.at_time(rays_of_twistors[:, None], core.mv.t & events)

    # checks: a ray's intersection with a time slice reconstructs the incident events, including
    # when a boost changes their time coordinates, and every sampled ray event sends its twistor to zero.
    np.testing.assert_allclose((recovered - events).kernel, 0, atol=TOLERANCE)
    np.testing.assert_allclose(core.incidence(core.point(rays), twistors[:, None]).kernel,
                               0, atol=TOLERANCE)
