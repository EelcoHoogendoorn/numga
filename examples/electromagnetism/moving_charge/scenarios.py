"""Moving charges: a soft charge sped up to near the speed of light and slowed down again, and a point
charge circling slowly and fast, shedding waves."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.electromagnetism.moving_charge import core

strength = 1.0
core_radius = 0.08
top_rapidity = 2.0
half_width = 1.5
half_height = 1.0
n_columns = 300
n_rows = 200
n_frames = 90
duration_ms = 60

orbit_charge = 1.0
orbit_radius = 1.0
# A slow and a fast orbit, as fractions of the speed of light, with the names of their animations.
orbit_speeds = np.array([0.5, 0.9])
orbit_names = ("slow", "fast")
orbit_half_width = 10.0
n_orbit_pixels = 300
n_orbit_frames = 60
# The coarser window the checks sample.
n_check_columns = 30
n_check_rows = 20


# --- math -----------------------------------------------------------------------------
def charge(rapidity: float) -> core.Charge:
    """The charge moving along x at the given rapidity, through the origin at lab time zero."""
    return core.Charge(np.array(strength), core_radius, (core.mv.xt * (rapidity / 2)).exp(), 0 * core.time)


def speed_up() -> Iterator[tuple[core.Antivector, core.Bivector, core.Antivector]]:
    """The lab's view at time zero as the charge speeds up: the events, the field, and the current."""
    events = core.grid(half_width, half_height, n_columns, n_rows)               # [rows, columns] Antivector
    # Slowing down retraces speeding up: the rise alone is half the frames.
    rapidities = top_rapidity * (1 - np.cos(2 * np.pi * np.arange(n_frames // 2 + 1) / n_frames)) / 2
    for rapidity in rapidities:
        moving = charge(rapidity)
        field = core.field(core.potential_gradient(moving, events))           # [rows, columns] Bivector
        current = core.current(core.field_gradient(moving, events))            # [rows, columns] Antivector
        yield events, field, current


def circling(orbit: core.Orbit) -> Iterator[tuple[core.Antivector, core.Bivector, core.Antivector]]:
    """Over one turn: the events in the orbit's plane at each lab time, the field there, and the
    charge."""
    plane = core.grid(orbit_half_width, orbit_half_width, n_orbit_pixels, n_orbit_pixels)   # [rows, columns] Antivector
    period = 2 * np.pi * orbit.radius / orbit.speed
    for time in period * np.arange(n_orbit_frames) / n_orbit_frames:
        events = plane + core.time * time                                    # [rows, columns] Antivector
        field = core.field(core.orbit_potential_gradient(orbit, events))     # [rows, columns] Bivector
        position, _, _ = core.worldline(orbit, core.mv.scalar([time]))       # [1] Antivector
        yield events, field, position


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation
    from examples.electromagnetism.moving_charge import render

    speeding_up = render.animate(speed_up())
    # Slowing down plays the rise back.
    save_animation(speeding_up + speeding_up[-2:0:-1], "moving_charge", duration_ms)
    for name, speed in zip(orbit_names, orbit_speeds):
        orbit = core.Orbit(orbit_charge, orbit_radius, speed)
        save_animation(render.animate_waves(list(circling(orbit))), f"radiating_charge_{name}", duration_ms)

    # checks
    # At the top speed the field's derivative vanishes, and the current is the softened density carried
    # along the charge's own time.
    moving = charge(top_rapidity)
    events = core.grid(half_width, half_height, n_check_columns, n_check_rows)  # [rows, columns] Antivector
    gradients = core.field_gradient(moving, events)
    np.testing.assert_allclose((core.Vector ^ gradients(core.Antivector)).trace(1, 2).kernel, 0.0, atol=1e-12)
    at_rest = moving.boost << events
    separations = at_rest - (core.mv.t & at_rest) * core.time
    softened = core_radius**2 + (separations | separations)
    density = 3 * strength * core_radius**2 / (4 * np.pi) / (softened * softened * softened.square_root())
    np.testing.assert_allclose((core.current(gradients) - density * (moving.boost >> core.time)).kernel, 0.0, atol=1e-9)


if __name__ == "__main__":
    main()
