"""Scenes for Jeffery orbits: a sphere, a prolate rod, and a triaxial ellipsoid tumbling in simple shear flow."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np

from numga import stack
from examples.fluids.jeffery import core

Point = core.Point
x, y, z = core.axes
# Principal axis ratios of the sphere, prolate rod, and triaxial ellipsoid, normalized to a common sphere radius.
axis_ratios = np.array([[1, 1, 1], [4, 1, 1], [4, 2, 1]], dtype=float)
equal_volume_radius = 1.3
semi_axes = equal_volume_radius * axis_ratios / np.prod(axis_ratios, axis=-1, keepdims=True) ** (1 / 3)
rod_aspect = axis_ratios[1, 0] / axis_ratios[1, 1]
# Ambient simple shear flow with velocity along x linear in height y.
shear_rate = 1.0
flow_gradient = shear_rate * x.dual() * (y & Point)                          # [] Point- <- Point-
jeffery_period = 2 * np.pi * (rod_aspect + 1 / rod_aspect) / shear_rate
# Initial polar tilts of the long axis from the vortex normal z toward the flow direction x.
initial_tilts = np.array([0.15, 0.4, 0.7, 1.0, 1.3])
shown_start = 2
n_periods = 2
n_steps_per_period = 120
n_steps_per_frame = 2
flow_heights = np.array([-3, -2, -1, 1, 2, 3], dtype=float)
frame_duration_ms = 40
gallery_pixels = 240


# --- math -----------------------------------------------------------------------------
def orbit_stream(n_steps: int) -> Iterator[core.Rotor]:
    """Yield every body's orientation rotor from every initial tilt across `n_steps` midpoint steps."""
    stretches = core.body_stretches(semi_axes)                               # [3] Point- <- Point-
    start_rotors = ((x ^ z) * (-(np.pi / 2 - initial_tilts) / 2)).exp()      # [5] Rotor+
    initial_rotors = start_rotors[None].broadcast_to((len(semi_axes), len(initial_tilts)))   # [3, 5] Rotor+
    dt = jeffery_period / n_steps_per_period
    return core.rotor_trajectory(initial_rotors, stretches[:, None], flow_gradient, dt, n_steps)


def shown_start_states(step_stream: Iterable[core.Rotor]) -> Iterator[tuple[core.Rotor, core.Point]]:
    """Yield strided single-start orientations and long-axis tips from a batch rotor stream."""
    for step_index, rotors in enumerate(step_stream):
        if step_index % n_steps_per_frame == 0:
            orientations = rotors[:, shown_start]                            # [3] Rotor+
            yield orientations, orientations >> (x.dual() * semi_axes[:, 0])


# --- plumbing -------------------------------------------------------------------------
def save_states(states: Iterable, animate, *arguments, name: str, duration_ms: int) -> list:
    """Save an animation drawn from `states` while collecting and returning the yielded states."""
    from examples.util.animation import save_animation

    kept = []

    def passing() -> Iterator:
        for state in states:
            kept.append(state)
            yield state

    save_animation(animate(*arguments, passing()), name, duration_ms)
    return kept


def main() -> None:
    from examples.util.animation import save_figure
    from examples.fluids.jeffery import render

    n_steps = n_periods * n_steps_per_period
    heights = y.dual() * flow_heights                                        # [6] Point-
    stretches = core.body_stretches(semi_axes)                               # [3] Point- <- Point-
    rotor_steps = save_states(
        orbit_stream(n_steps),
        lambda s, h, v, px, steps: render.animate_tumbling(s, h, v, px, shown_start_states(steps)),
        stretches,
        heights,
        flow_gradient(heights),
        gallery_pixels,
        name="jeffery_tumbling",
        duration_ms=frame_duration_ms,
    )
    rotors = stack(rotor_steps)                                              # [n_steps + 1, 3, 5] Rotor+
    trails = rotors >> x.dual()                                              # [n_steps + 1, 3, 5] Point-
    save_figure(render.draw_orbits(trails), "jeffery_orbits")

    # checks
    assert ((rotors.scalar_norm_squared() - 1).abs() < 1e-11).all()
    sphere_shapes = core.body_shapes(stretches[0], rotors[-1, 0])            # [5] Point- <- Plane+
    assert ((core.jeffery_spin(sphere_shapes, flow_gradient) - core.flow_curl(flow_gradient) / 2).norm() < 1e-12).all()
    rod_trails = trails[:, 1]                                                # [n_steps + 1, 5] Point-
    assert ((rod_trails[n_steps_per_period] - rod_trails[0]).norm() < 2e-2).all()
    flow_coord, gradient_coord, vortex_coord = x & rod_trails, y & rod_trails, z & rod_trails   # [n_steps + 1, 5] Scalar+, Scalar+, Scalar+
    orbit_constant = (flow_coord.squared() + rod_aspect**2 * gradient_coord.squared()) / (rod_aspect**2 * vortex_coord.squared())
    assert ((orbit_constant / orbit_constant[0] - 1).abs() < 2e-2).all()
    assert np.mean(flow_coord.abs() > 0.8 * np.sin(initial_tilts)) > 0.5


def pinned() -> None:
    """Regenerate the repository-tracked gallery animation of the three bodies tumbling in shear flow."""
    from examples.util.animation import pin_animation
    from examples.fluids.jeffery import render

    n_steps = n_periods * n_steps_per_period
    heights = y.dual() * flow_heights                                        # [6] Point-
    pin_animation(
        render.animate_tumbling(core.body_stretches(semi_axes), heights, flow_gradient(heights), gallery_pixels,
                                shown_start_states(orbit_stream(n_steps))),
        "jeffery",
        frame_duration_ms,
    )


if __name__ == "__main__":
    main()


