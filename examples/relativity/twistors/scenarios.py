"""A light ray recovered from its twistor, linked flow lines, and a propagating electromagnetic pulse."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.relativity.twistors import core

mv = core.mv
# The boosts that carry the incidence along, the first at rest, and the turn from the flow lines to the electric lines.
RAPIDITIES = np.array([0.0, 0.8])
BOOST = (mv.xt * (RAPIDITIES / 2)).exp()                                     # [cases] Rotor
ELECTRIC_TURN = (mv.xy * (np.pi / 4)).exp()


# --- math -----------------------------------------------------------------------------
def incidence() -> tuple[core.Twistor, core.Event, core.Event]:
    """The twistor of the light ray through two null-separated events, the events and the reconstructed
    ray, once per boost."""
    emission, light_direction, separation = mv.x * -0.6 + mv.z * 0.3, mv.t + (mv.x + mv.z) / np.sqrt(2), 1.4
    margin, ray_samples = 0.6, 120
    events = stack([emission, emission + light_direction * separation])       # [events] Event
    points = core.point(events)                                              # [events] Vector
    twistor = core.through(points[0], points[1])                              # [] Twistor

    # Transform the twistor with the spinor action and the events with the sandwich.
    twistors = core.REPRESENTATIONS(BOOST, twistor)                           # [cases] Twistor
    planes = core.RAY(twistors, twistors)                                     # [cases] Bivector
    moved_points = BOOST[:, None] >> points                                   # [cases, events] Vector
    moved_events = moved_points.cast(core.Event) / -(core.INFINITY | moved_points)   # [cases, events] Event
    fractions = np.linspace(-margin, 1 + margin, ray_samples)
    event_times = -(mv.t | moved_events)                                     # [cases, events] Scalar
    ray_times = event_times[:, :1] + (event_times[:, 1:] - event_times[:, :1]) * fractions
    rays = core.at_time(planes[:, None], ray_times)                           # [cases, ray_samples] Event
    return twistors, moved_events, rays


def congruence() -> core.Spatial:
    """Instantaneous flow lines of the Robinson congruence at time zero."""
    polars, per_circle, fibre_samples = np.pi * np.array([0.5, 0.65, 0.8]), 10, 240
    return core.fibres(polars, per_circle, fibre_samples)


def electric_lines() -> core.Spatial:
    """The Hopfion's electric field lines at time zero: the flow lines, a quarter turn in the xy plane.
    Its magnetic lines are the flow lines a quarter turn in the yz plane, and look alike."""
    return ELECTRIC_TURN >> congruence()


def propagation(times: np.ndarray) -> Iterator[core.Event]:
    """Electric field lines transported along straight null rays, one set per instant."""
    electric = electric_lines()                                               # [polars, per_circle, fibre_samples + 1] Spatial
    rays = core.robinson(electric)                                            # [polars, per_circle, fibre_samples + 1] Event
    # The null field's lines are carried by its energy flow: each material point follows one
    # straight light ray, even as the lines deform.
    for time in times:
        yield electric + rays * time


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation, save_figure
    from examples.relativity.twistors import render

    frames, frame_ms = 48, 80
    twistors, events, rays = incidence()
    save_figure(render.draw_incidence(events, rays), "twistor_incidence")
    save_figure(render.draw_congruence(congruence()), "twistor_robinson")
    save_figure(render.draw_fields(electric_lines()), "twistor_hopfion")
    save_animation(render.animate_fields(propagation(np.linspace(-1.25, 1.25, frames))), "twistor_hopfion", frame_ms)

    # --- checks
    # The recovered ray meets the events it came from, at rest and boosted.
    planes = core.RAY(twistors, twistors)                                     # [cases] Bivector
    np.testing.assert_allclose((core.at_time(planes[:, None], -(events | mv.t)) - events).kernel, 0.0, atol=1e-11)
    np.testing.assert_allclose(core.REPRESENTATIONS(core.point(rays), twistors[:, None]).kernel, 0.0, atol=1e-11)
    # The field is null, and its energy flows along the Robinson ray through each point of its lines.
    lines = electric_lines()
    field = core.hopfion(lines)
    electric, magnetic = field | mv.t, (mv.xyzt * field) | mv.t
    energy = (electric.squared() + magnetic.squared()) / 2
    flow = -core.SPATIAL_VOLUME * (electric ^ magnetic)
    np.testing.assert_allclose(field.squared().kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose((mv.t + flow / energy - core.robinson(lines)).kernel, 0.0, atol=1e-10)


if __name__ == "__main__":
    main()
