"""Twistors and linked light: a light ray recovered from its twistor, linked flow lines, and a propagating
electromagnetic pulse."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.relativity.twistors import core

mv = core.mv
# The boosts that carry the incidence along, the first at rest, and the turn from the flow lines to the electric lines.
RAPIDITIES = np.array([0.0, 0.8])
boost = ((mv.x.dual() & core.time).dual() * (RAPIDITIES / 2)).exp()           # [cases] Rotor+
electric_turn = (mv.xy * (np.pi / 4)).exp()


# --- math -----------------------------------------------------------------------------
def incidence() -> tuple[core.Twistor, core.Event, core.Event]:
    """The twistor of the light ray through two null-separated events, the events and the reconstructed
    ray, once per boost."""
    emission = mv.x.dual() * -0.6 + mv.z.dual() * 0.3                         # [] Event-
    light_direction = core.time + (mv.x.dual() + mv.z.dual()) / np.sqrt(2)    # [] Event-
    separation = 1.4
    margin, n_ray_samples = 0.6, 120
    events = stack([emission, emission + light_direction * separation])       # [events] Event-
    points = core.point(events)                                               # [events] Point-
    twistor = core.through(points[0], points[1])                              # [] Twistor-

    # Transform the twistor with the spinor action and the events with the sandwich.
    twistors = core.representations(boost, twistor)                           # [cases] Twistor-
    light_rays = core.ray_readout(twistors, twistors)                                 # [cases] Ray-
    moved_points = boost[:, None] >> points                                   # [cases, events] Point-
    moved_events = moved_points.cast(core.Event) / (core.horizon & moved_points)   # [cases, events] Event-
    fractions = np.linspace(-margin, 1 + margin, n_ray_samples)
    event_times = mv.t & moved_events                                         # [cases, events] Scalar+
    ray_times = event_times[:, :1] + (event_times[:, 1:] - event_times[:, :1]) * fractions
    rays = core.at_time(light_rays[:, None], ray_times)                       # [cases, ray samples] Event-
    return twistors, moved_events, rays


def congruence() -> core.Spatial:
    """Instantaneous flow lines of the Robinson congruence at time zero."""
    polars, n_per_circle, n_fibre_samples = np.pi * np.array([0.5, 0.65, 0.8]), 10, 240
    return core.fibres(polars, n_per_circle, n_fibre_samples)


def electric_lines() -> core.Spatial:
    """The Hopfion's electric field lines at time zero: the flow lines, a quarter turn in the xy plane.
    Its magnetic lines are the flow lines a quarter turn in the yz plane, and look alike."""
    return electric_turn >> congruence()


def propagation(times: np.ndarray) -> Iterator[core.Event]:
    """Electric field lines transported along straight null rays, one set per instant."""
    electric = electric_lines()                                               # [polars, per circle, fibre samples] Spatial-
    rays = core.robinson(electric)                                            # [polars, per circle, fibre samples] Event-
    # The null field's lines are carried by its energy flow: each material point follows one
    # straight light ray, even as the lines deform.
    for time in times:
        yield electric + rays * time


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_figure
    from examples.relativity.twistors import render

    twistors, events, rays = incidence()
    save_figure(render.draw_incidence(events, rays), "twistor_incidence")
    save_figure(render.draw_congruence(congruence()), "twistor_robinson")
    save_figure(render.draw_fields(electric_lines()), "twistor_hopfion")

    # --- checks
    # The recovered ray meets the events it came from, at rest and boosted.
    light_rays = core.ray_readout(twistors, twistors)                                 # [cases] Ray-
    np.testing.assert_allclose((core.at_time(light_rays[:, None], mv.t & events) - events).kernel, 0.0, atol=1e-11)
    np.testing.assert_allclose(core.incidence(core.point(rays), twistors[:, None]).kernel, 0.0, atol=1e-11)
    # The field is null, and its energy flows along the Robinson ray through each point of its lines: the field's
    # flux along the observer's time, its sandwich of `core.time`, taken at unit time.
    lines = electric_lines()
    field = core.hopfion(lines)
    flux = field >> core.time
    np.testing.assert_allclose(field.squared().kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose((flux / (mv.t & flux) - core.robinson(lines)).kernel, 0.0, atol=1e-10)


def pinned() -> None:
    """The pictures the repository tracks: the Hopfion's linked electric field lines carried along its light rays, for the README and the gallery."""
    from examples.util.animation import pin_animation
    from examples.relativity.twistors import render

    # The frames, their duration in milliseconds, and the colours they are drawn in.
    n_frames, duration_ms, n_colours = 96, 40, 32
    pin_animation(render.animate_fields(propagation(np.linspace(-1.25, 1.25, n_frames))), "twistor_hopfion", duration_ms,
                  colors=n_colours)


if __name__ == "__main__":
    main()
