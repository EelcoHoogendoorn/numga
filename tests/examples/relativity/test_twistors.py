"""Twistor incidence, conformal covariance, and the straight rays of the Robinson congruence."""

import numpy as np

from examples.relativity.twistors import core


TOLERANCE = 1e-11
SAMPLES = 24


def conformal_rotors(rng: np.random.Generator) -> core.Full:
    """Random rotations, boosts, translations and special conformal transformations, composed."""
    mv = core.mv
    translations = mv(core.Event, rng.normal(size=(SAMPLES, 4)) * 0.2)
    special = mv(core.Event, rng.normal(size=(SAMPLES, 4)) * 0.2)
    return (
        (mv.xy * rng.normal(size=SAMPLES) * 0.2).exp()
        * (mv.xt * rng.normal(size=SAMPLES) * 0.2).exp()
        * ((translations ^ core.INFINITY) * -0.5).exp()
        * ((special ^ core.ORIGIN) * -0.5).exp()
    )


def test_clifford_action_and_hermitian_pairing_follow_conformal_rotors():
    rng = np.random.default_rng(821)
    mv = core.mv
    first = mv(core.Full, rng.normal(size=(SAMPLES, 64)) * 0.2)
    second = mv(core.Full, rng.normal(size=(SAMPLES, 64)) * 0.2)
    states = mv(core.Twistor, rng.normal(size=(SAMPLES, 8)))
    partners = mv(core.Twistor, rng.normal(size=(SAMPLES, 8)))
    rotors = conformal_rotors(rng)
    moved = core.REPRESENTATIONS(rotors, states)
    moved_partners = core.REPRESENTATIONS(rotors, partners)

    # Clifford multiplication acts by composition; even actions commute with the
    # volume and preserve both pairings.
    composition = core.REPRESENTATIONS(first)(core.REPRESENTATIONS(second))
    product = core.REPRESENTATIONS(first * second)
    np.testing.assert_allclose((composition - product).kernel, 0, atol=TOLERANCE)
    np.testing.assert_allclose((core.VOLUME(core.VOLUME(states)) + states).kernel,
                               0, atol=TOLERANCE)
    np.testing.assert_allclose(
        (core.VOLUME(moved) - core.REPRESENTATIONS(rotors, core.VOLUME(states))).kernel,
        0, atol=TOLERANCE,
    )
    np.testing.assert_allclose(
        (core.PAIRING(moved, moved_partners) - core.PAIRING(states, partners)).kernel,
        0, atol=TOLERANCE,
    )
    np.testing.assert_allclose(
        (core.PAIRING(core.VOLUME(moved), moved_partners)
         - core.PAIRING(core.VOLUME(states), partners)).kernel,
        0, atol=TOLERANCE,
    )


def test_shared_twistor_recovers_a_null_ray_and_transforms_with_its_events():
    rng = np.random.default_rng(822)
    mv = core.mv
    events = mv(core.Event, rng.normal(size=(SAMPLES, 4)))
    directions = mv.t + mv(core.Spatial, rng.normal(size=(SAMPLES, 3))).normalized()
    separation = rng.uniform(0.2, 1.5, size=SAMPLES)
    first = core.point(events)
    second = core.point(events + directions * separation)
    twistors = core.through(first, second)
    planes = core.RAY(twistors, twistors)
    phases = (mv.xyztuv * rng.normal(size=SAMPLES)).exp()
    rotors = conformal_rotors(rng)
    moved = core.REPRESENTATIONS(rotors, twistors)

    # The two events have a shared null twistor. Its bilinear ray is the null
    # plane through them, and turning it by the volume leaves its direction alone.
    np.testing.assert_allclose(first.squared().kernel, 0, atol=TOLERANCE)
    np.testing.assert_allclose(second.squared().kernel, 0, atol=TOLERANCE)
    np.testing.assert_allclose(core.REPRESENTATIONS(first, twistors).kernel, 0, atol=TOLERANCE)
    np.testing.assert_allclose(core.REPRESENTATIONS(second, twistors).kernel, 0, atol=TOLERANCE)
    np.testing.assert_allclose(core.PAIRING(twistors, twistors).kernel, 0, atol=TOLERANCE)
    np.testing.assert_allclose((first ^ planes).kernel, 0, atol=TOLERANCE)
    np.testing.assert_allclose((second ^ planes).kernel, 0, atol=TOLERANCE)
    np.testing.assert_allclose(planes.squared().kernel, 0, atol=TOLERANCE)
    np.testing.assert_allclose((core.direction(twistors) - directions).kernel,
                               0, atol=TOLERANCE)
    np.testing.assert_allclose(
        (core.direction(core.REPRESENTATIONS(phases, twistors)) - directions).kernel,
        0, atol=TOLERANCE,
    )
    np.testing.assert_allclose(core.REPRESENTATIONS(rotors >> first, moved).kernel,
                               0, atol=TOLERANCE)
    np.testing.assert_allclose(core.REPRESENTATIONS(rotors >> second, moved).kernel,
                               0, atol=TOLERANCE)
    np.testing.assert_allclose(
        (core.RAY(moved, moved) - (rotors >> planes)).kernel, 0, atol=TOLERANCE,
    )


def test_robinson_twistors_stay_orthogonal_along_straight_future_null_rays():
    rng = np.random.default_rng(823)
    events = core.mv(core.Event, rng.normal(size=(SAMPLES, 4)))
    twistors = core.REPRESENTATIONS(core.point(events), core.ROBINSON)
    directions = core.robinson(events)
    times = np.linspace(-1.0, 1.0, 9)
    along_rays = events[:, None] + directions[:, None] * times

    # The event's selected twistor pairs to zero with the fixed twistor and with
    # its turn by the volume. Every event along its straight light ray selects the
    # same propagation direction and remains incident with the initial twistor.
    np.testing.assert_allclose(core.PAIRING(core.ROBINSON, twistors).kernel,
                               0, atol=TOLERANCE)
    np.testing.assert_allclose(core.PAIRING(core.VOLUME(core.ROBINSON), twistors).kernel,
                               0, atol=TOLERANCE)
    np.testing.assert_allclose(core.PAIRING(twistors, twistors).kernel, 0, atol=TOLERANCE)
    np.testing.assert_allclose(directions.squared().kernel, 0, atol=TOLERANCE)
    np.testing.assert_allclose(((core.mv.t | directions) + 1).kernel, 0, atol=TOLERANCE)
    np.testing.assert_allclose((core.robinson(along_rays) - directions[:, None]).kernel,
                               0, atol=TOLERANCE)
    np.testing.assert_allclose(core.REPRESENTATIONS(core.point(along_rays), twistors[:, None]).kernel,
                               0, atol=TOLERANCE)


POLARS = np.pi * np.array([0.4, 0.65])
PER_CIRCLE = 3
FIBRE_SAMPLES = 192
TIMES = np.array([-0.8, 0.0, 0.8])
# Centred differences of sampled curves resolve their tangents to about this accuracy.
TANGENT_TOLERANCE = 0.003


def test_flow_lines_follow_the_robinson_directions():
    curves = core.fibres(POLARS, PER_CIRCLE, FIBRE_SAMPLES)
    tangent = (curves[..., 2:] - curves[..., :-2]).normalized()
    flow = core.robinson(curves[..., 1:-1]).cast(core.Spatial)

    # checks: the sampled circles run along the spatial directions of the congruence.
    np.testing.assert_allclose((tangent ^ flow).kernel, 0, atol=TANGENT_TOLERANCE)


def test_field_lines_follow_their_fields_along_the_energy_flow_rays():
    from examples.relativity.twistors import scenarios

    curves = core.fibres(POLARS, PER_CIRCLE, FIBRE_SAMPLES)
    electric = scenarios.ELECTRIC_TURN >> curves
    magnetic = (core.mv.yz * (np.pi / 4)).exp() >> curves
    times = TIMES[:, None, None, None]
    electric_events = electric + core.robinson(electric) * times
    magnetic_events = magnetic + core.robinson(magnetic) * times
    electric_field = core.hopfion(electric_events) | core.mv.t
    magnetic_field = (core.mv.xyzt * core.hopfion(magnetic_events)) | core.mv.t
    electric_tangent = (electric_events[..., 2:] - electric_events[..., :-2]).normalized()
    magnetic_tangent = (magnetic_events[..., 2:] - magnetic_events[..., :-2]).normalized()

    # checks: after their material points have followed the straight energy-flow rays, the electric
    # and the magnetic lines still run along their fields.
    np.testing.assert_allclose(
        (electric_tangent ^ electric_field[..., 1:-1].normalized()).kernel, 0, atol=TANGENT_TOLERANCE,
    )
    np.testing.assert_allclose(
        (magnetic_tangent ^ magnetic_field[..., 1:-1].normalized()).kernel, 0, atol=TANGENT_TOLERANCE,
    )


def test_two_fibres_link_once():
    curves = core.fibres(POLARS, PER_CIRCLE, FIBRE_SAMPLES)
    first, second = curves[0, 0], curves[0, 1]
    first_steps, second_steps = first[1:] - first[:-1], second[1:] - second[:-1]
    first_midpoints, second_midpoints = (first[1:] + first[:-1]) / 2, (second[1:] + second[:-1]) / 2
    offsets = first_midpoints[:, None] - second_midpoints[None, :]
    distance_squared = offsets.scalar_norm_squared()
    volume = offsets ^ first_steps[:, None] ^ second_steps[None, :]
    linking = (core.SPATIAL_VOLUME.scalar_product(volume)
               / (distance_squared * distance_squared.square_root())).sum() / (4 * np.pi)

    # checks: the Gauss integral counts one linking of two distinct closed fibres.
    np.testing.assert_allclose(np.abs(linking.kernel), 1, rtol=0, atol=0.0001)


def test_incidence_rays_meet_their_events_at_rest_and_boosted():
    from examples.relativity.twistors import scenarios

    twistors, events, rays = scenarios.incidence()
    planes = core.RAY(twistors, twistors)
    recovered = core.at_time(planes[:, None], -(events | core.mv.t))

    # checks: a ray's intersection with a time slice reconstructs the incident events, including
    # when a boost changes their time coordinates, and every sampled ray event sends its twistor to zero.
    np.testing.assert_allclose((recovered - events).kernel, 0, atol=TOLERANCE)
    np.testing.assert_allclose(core.REPRESENTATIONS(core.point(rays), twistors[:, None]).kernel,
                               0, atol=TOLERANCE)


def test_scenes_draw():
    import matplotlib.pyplot as plt

    from examples.relativity.twistors import render, scenarios

    curves = core.fibres(POLARS, PER_CIRCLE, FIBRE_SAMPLES)
    _, events, rays = scenarios.incidence()
    for figure in (render.draw_incidence(events, rays), render.draw_congruence(curves),
                   render.draw_fields(curves)):
        plt.close(figure)
    render.animate_fields(scenarios.propagation(TIMES))
