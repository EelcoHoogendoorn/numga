"""A light ray recovered from its twistor, linked flow lines, and a propagating electromagnetic pulse."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.relativity.twistors import core

mv = core.mv
# The boost that carries the incidence along, and the turn from the flow lines to the electric lines.
RAPIDITY = 0.8
ELECTRIC_TURN = (mv.xy * (np.pi / 4)).exp()


# --- math -----------------------------------------------------------------------------
def incidence() -> tuple[core.Event, core.Event, core.Event, core.Event]:
    """Two null-separated events, their reconstructed light ray, and their Lorentz-boosted images."""
    emission, light_direction, separation = mv.x * -0.6 + mv.z * 0.3, mv.t + (mv.x + mv.z) / np.sqrt(2), 1.4
    margin, ray_samples = 0.6, 120
    events = stack([emission, emission + light_direction * separation])       # [events] Event
    points = core.point(events)                                              # [events] Vector
    twistor = core.through(points[0], points[1])                              # [] Twistor
    plane = core.RAY(twistor, twistor)                                        # [] Bivector
    fractions = np.linspace(-margin, 1 + margin, ray_samples)
    event_times = -(mv.t | events)
    ray_times = event_times[0] + (event_times[1] - event_times[0]) * fractions
    ray = core.at_time(plane, ray_times)                                      # [ray_samples] Event

    # Transform the twistor with the spinor action and the events with the sandwich.
    rotor = (mv.xt * (RAPIDITY / 2)).exp()
    moved_twistor = core.REPRESENTATIONS(rotor, twistor)
    moved_plane = core.RAY(moved_twistor, moved_twistor)
    moved_points = rotor >> points
    moved_events = moved_points.cast(core.Event) / -(core.INFINITY | moved_points)
    moved_event_times = -(mv.t | moved_events)
    moved_ray_times = moved_event_times[0] + (moved_event_times[1] - moved_event_times[0]) * fractions
    moved_ray = core.at_time(moved_plane, moved_ray_times)
    return events, ray, moved_events, moved_ray


def congruence() -> core.Spatial:
    """Instantaneous flow lines of the Robinson congruence at time zero."""
    polars, per_circle, fibre_samples = np.pi * np.array([0.5, 0.65, 0.8]), 10, 240
    return core.fibres(polars, per_circle, fibre_samples)


def electric_lines() -> core.Spatial:
    """The Hopfion's electric field lines at time zero: the flow lines, turned. Its magnetic lines
    are the same family turned the other way, and look alike."""
    return ELECTRIC_TURN >> congruence()


def propagation(times: np.ndarray) -> Iterator[core.Event]:
    """Electric field lines transported along straight null rays, one set per instant."""
    electric = electric_lines()                                               # [curves, fibre_samples + 1] Spatial
    rays = core.robinson(electric)                                            # [curves, fibre_samples + 1] Event
    # The null field's lines are carried by its energy flow: each material point follows one
    # straight light ray, even as the lines deform.
    for time in times:
        yield electric + rays * time


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation, save_figure
    from examples.relativity.twistors import render

    frames, frame_ms = 48, 80
    events, ray, moved_events, moved_ray = incidence()
    save_figure(render.draw_incidence(events, ray, moved_events, moved_ray), "twistor_incidence")
    save_figure(render.draw_congruence(congruence()), "twistor_robinson")
    save_figure(render.draw_fields(electric_lines()), "twistor_hopfion")
    save_animation(render.animate_fields(propagation(np.linspace(-1.25, 1.25, frames))), "twistor_hopfion", frame_ms)

    # --- checks
    # The recovered ray meets the events it came from, before and after the boost.
    points = core.point(events)
    twistor = core.through(points[0], points[1])                              # [] Twistor
    moved_twistor = core.REPRESENTATIONS((mv.xt * (RAPIDITY / 2)).exp(), twistor)   # [] Twistor
    np.testing.assert_allclose((core.at_time(core.RAY(twistor, twistor), -(events | mv.t)) - events).kernel, 0.0, atol=1e-11)
    np.testing.assert_allclose(core.REPRESENTATIONS(core.point(moved_ray), moved_twistor).kernel, 0.0, atol=1e-11)
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
