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
BOOST = ((mv.x.dual() & core.TIME).dual() * (RAPIDITIES / 2)).exp()           # [cases] Rotor+
ELECTRIC_TURN = (mv.xy * (np.pi / 4)).exp()


# --- math -----------------------------------------------------------------------------
def incidence() -> tuple[core.Twistor, core.Event, core.Event]:
    """The twistor of the light ray through two null-separated events, the events and the reconstructed
    ray, once per boost."""
    emission = mv.x.dual() * -0.6 + mv.z.dual() * 0.3                         # [] Event-
    light_direction = core.TIME + (mv.x.dual() + mv.z.dual()) / np.sqrt(2)    # [] Event-
    separation = 1.4
    margin, n_ray_samples = 0.6, 120
    events = stack([emission, emission + light_direction * separation])       # [events] Event-
    points = core.point(events)                                               # [events] Point-
    twistor = core.through(points[0], points[1])                              # [] Twistor-

    # Transform the twistor with the spinor action and the events with the sandwich.
    twistors = core.REPRESENTATIONS(BOOST, twistor)                           # [cases] Twistor-
    light_rays = core.RAY(twistors, twistors)                                 # [cases] Ray-
    moved_points = BOOST[:, None] >> points                                   # [cases, events] Point-
    moved_events = moved_points.cast(core.Event) / (core.HORIZON & moved_points)   # [cases, events] Event-
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
    return ELECTRIC_TURN >> congruence()


# README picture: this makes plots/twistor_hopfion.gif, tuned as a visual for the README. Don't change it; to
# experiment, copy it.
def propagation(times: np.ndarray) -> Iterator[core.Event]:
    """Electric field lines transported along straight null rays, one set per instant."""
    electric = electric_lines()                                               # [polars, per circle, fibre samples + 1] Spatial-
    rays = core.robinson(electric)                                            # [polars, per circle, fibre samples + 1] Event-
    # The null field's lines are carried by its energy flow: each material point follows one
    # straight light ray, even as the lines deform.
    for time in times:
        yield electric + rays * time


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.relativity.twistors import render

    frames, frame_ms = 48, 80
    twistors, events, rays = incidence()
    save_figure(render.draw_incidence(events, rays), "twistor_incidence")
    save_figure(render.draw_congruence(congruence()), "twistor_robinson")
    save_figure(render.draw_fields(electric_lines()), "twistor_hopfion")
    # README picture: this makes plots/twistor_hopfion.gif, tuned as a visual for the README. Don't change it; to
    # experiment, copy it.
    save_animation(render.animate_fields(propagation(np.linspace(-1.25, 1.25, frames))), "twistor_hopfion", frame_ms)

    # --- checks
    # The recovered ray meets the events it came from, at rest and boosted.
    light_rays = core.RAY(twistors, twistors)                                 # [cases] Ray-
    np.testing.assert_allclose((core.at_time(light_rays[:, None], mv.t & events) - events).kernel, 0.0, atol=1e-11)
    np.testing.assert_allclose(core.INCIDENCE(core.point(rays), twistors[:, None]).kernel, 0.0, atol=1e-11)
    # The field is null, and its energy flows along the Robinson ray through each point of its lines: the field's
    # flux along the observer's time, its sandwich of `core.TIME`, taken at unit time.
    lines = electric_lines()
    field = core.hopfion(lines)
    flux = field >> core.TIME
    np.testing.assert_allclose(field.squared().kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose((flux / (mv.t & flux) - core.robinson(lines)).kernel, 0.0, atol=1e-10)


if __name__ == "__main__":
    main()
