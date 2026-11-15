"""Scenarios for twistors and linked light reconstruct a boosted light ray from its twistor, sample the Robinson congruence, and propagate the Hopfion's linked electric field lines."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.relativity.twistors import core

mv = core.mv
# Emission event, null propagation direction, and separation along the light ray.
emission_coords = [-0.6, 0.0, 0.3, 0.0]
separation = 1.4
# Rest and boosted rapidities along x, plus ray sampling count and fractional time margin.
rapidities = np.array([0.0, 0.8])
n_boosts = len(rapidities)
n_ray_samples = 120
ray_margin = 0.6
boost = ((mv.x.dual() & core.time).dual() * (rapidities / 2)).exp()           # [n_boosts] Even+
# Nested torus polar angles, circle count per torus, and sample count per fibre.
polar_angles = np.pi * np.array([0.5, 0.65, 0.8])
n_tori = len(polar_angles)
n_circles = 10
n_fibre_samples = 240
# Quarter-turn rotor in the xy plane carrying Robinson flow lines onto initial electric field lines.
electric_turn = (mv.xy * (np.pi / 4)).exp()                                   # [] Even+
# Probe spatial offset and animation frame settings.
probe_coords = [0.4, 0.0, 0.2, 0.0]
n_frames = 96
frame_duration_ms = 40
time_span = 1.25


# --- math -----------------------------------------------------------------------------
def ray_incidence() -> tuple[core.Twistor, core.Event, core.Event]:
    """Reconstructing the light ray through two null-separated events at rest and under a Lorentz boost gives the twistors, events, and sampled rays (`[n_boosts, ...]`)."""
    emission = mv(core.Event.derive.dual, emission_coords).dual()             # [] Event-
    light_direction = core.time + (mv.x.dual() + mv.z.dual()) / np.sqrt(2)    # [] Event-
    events = stack([emission, emission + light_direction * separation])       # [2] Event-
    points = core.conformal_point(events)                                     # [2] Point-
    twistor = core.ray_twistor(points[0], points[1])                          # [] Twistor-

    # Act on the twistor from one side and on conformal points by the rotor sandwich.
    twistors = core.representations(boost, twistor)                           # [n_boosts] Twistor-
    light_rays = core.ray_readout(twistors, twistors)                         # [n_boosts] Ray-
    moved_points = boost[:, None] >> points                                   # [n_boosts, 2] Point-
    moved_events = moved_points.cast(core.Event) / (core.horizon & moved_points)  # [n_boosts, 2] Event-
    fractions = np.linspace(-ray_margin, 1 + ray_margin, n_ray_samples)       # [n_ray_samples]
    event_times = mv.t & moved_events                                         # [n_boosts, 2] Scalar+
    ray_times = event_times[:, :1] + (event_times[:, 1:] - event_times[:, :1]) * fractions  # [n_boosts, n_ray_samples] Scalar+
    rays = core.ray_events(light_rays[:, None], ray_times)                    # [n_boosts, n_ray_samples] Event-
    return twistors, moved_events, rays


def congruence_lines() -> core.Spatial:
    """Instantaneous spatial flow lines (`[n_tori, n_circles, n_fibre_samples] Spatial-`) follow the Robinson congruence at time zero."""
    return core.hopf_fibres(polar_angles, n_circles, n_fibre_samples)         # [n_tori, n_circles, n_fibre_samples] Spatial-


def electric_lines() -> core.Spatial:
    """Turning the Robinson flow lines by a quarter turn in `xy` produces the Hopfion's initial electric field lines (`[n_tori, n_circles, n_fibre_samples] Spatial-`)."""
    flow_lines = congruence_lines()                                           # [n_tori, n_circles, n_fibre_samples] Spatial-
    return electric_turn >> flow_lines                                        # [n_tori, n_circles, n_fibre_samples] Spatial-


def field_line_states(times: np.ndarray) -> Iterator[core.Event]:
    """Yields the Hopfion's electric field lines (`[n_tori, n_circles, n_fibre_samples] Event-`) carried along straight Robinson light rays at `times`."""
    return core.propagated_lines(electric_lines(), times)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    """Renders the twistor incidence, Robinson congruence, and Hopfion field figures and animation, then checks their algebraic identities."""
    from examples.util.animation import save_animation, save_figure
    from examples.relativity.twistors import render

    twistors, events, rays = ray_incidence()                                  # [n_boosts] Twistor-, [n_boosts, 2] Event-, [n_boosts, n_ray_samples] Event-
    flow_lines = congruence_lines()                                           # [n_tori, n_circles, n_fibre_samples] Spatial-
    initial_electric = electric_turn >> flow_lines                            # [n_tori, n_circles, n_fibre_samples] Spatial-
    frame_times = np.linspace(-time_span, time_span, n_frames)                # [n_frames]

    save_figure(render.incidence_figure(events, rays), "twistor_incidence")
    save_figure(render.congruence_figure(flow_lines), "twistor_robinson")
    save_figure(render.fields_figure(initial_electric), "twistor_hopfion")

    streamed = []

    def recorded_states() -> Iterator[core.Event]:
        for state in core.propagated_lines(initial_electric, frame_times):
            streamed.append(state)
            yield state

    save_animation(render.fields_animation(recorded_states()), "twistor_hopfion", frame_duration_ms, colors=32)

    # checks
    light_rays = core.ray_readout(twistors, twistors)                         # [n_boosts] Ray-
    assert ((core.ray_events(light_rays[:, None], mv.t & events) - events).scalar_norm_squared() < 1e-18).all()
    assert (core.incidence(core.conformal_point(rays), twistors[:, None]).scalar_norm_squared() < 1e-18).all()
    field = core.hopfion_field(initial_electric)                              # [n_tori, n_circles, n_fibre_samples] Field+
    flux = field >> core.time                                                 # [n_tori, n_circles, n_fibre_samples] Event-
    assert (field.squared().scalar_norm_squared() < 1e-18).all()
    assert ((flux / (mv.t & flux) - core.robinson_direction(initial_electric)).scalar_norm_squared() < 1e-16).all()
    assert len(streamed) == n_frames


def pinned() -> None:
    """Pins the gallery animation of the Hopfion's linked electric field lines propagated along Robinson light rays."""
    from examples.util.animation import pin_animation
    from examples.relativity.twistors import render

    pin_animation(
        render.fields_animation(field_line_states(np.linspace(-time_span, time_span, n_frames))),
        "twistor_hopfion",
        frame_duration_ms,
        colors=32,
    )


if __name__ == "__main__":
    main()
