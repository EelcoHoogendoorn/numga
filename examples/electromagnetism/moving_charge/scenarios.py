"""Moving charges: a soft charge sped up to near the speed of light and slowed down again, and a point
charge circling slowly and fast, shedding waves."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.electromagnetism.moving_charge import core

strength = 1.0
core_radius = 0.08
top_rapidity = 2.0
# The lab at time zero, an upright window about the charge, which runs across it, so that its field gathers into an
# upright band; an event for each pixel.
half_width = 0.45
half_height = 0.9
n_columns = 150
n_rows = 300
n_frames = 136
duration_ms = 40

orbit_charge = 1.0
orbit_radius = 1.0
# A slow and a fast orbit, as fractions of the speed of light, with the names of their animations; their pictures'
# pixels a side, the frames of a turn, and the colours of their palettes.
orbit_speeds = np.array([0.5, 0.9])
orbit_names = ("slow", "fast")
orbit_half_width = 10.0
n_orbit_pixels = 300
n_orbit_frames = 90
n_orbit_colours = 32
# The coarser window the checks sample.
n_check_columns = 30
n_check_rows = 20


# --- math -----------------------------------------------------------------------------
def charge(rapidity: float) -> core.Charge:
    """The charge moving along x at the given rapidity, through the origin at lab time zero."""
    return core.Charge(np.array(strength), core_radius, (core.mv.xt * (rapidity / 2)).exp(), 0 * core.time)


def speed_up() -> Iterator[tuple[core.Antivector, core.Bivector, core.Antivector]]:
    """The lab's view at time zero as the charge speeds up: the events, the field, and the current."""
    events = core.grid(half_width, half_height, n_columns, n_rows)               # [rows, columns] Antivector-
    # Slowing down retraces speeding up: the rise alone is half the frames.
    rapidities = top_rapidity * (1 - np.cos(2 * np.pi * np.arange(n_frames // 2 + 1) / n_frames)) / 2
    for rapidity in rapidities:
        moving = charge(rapidity)
        field = core.field(core.potential_gradient(moving, events))           # [rows, columns] Bivector+
        current = core.current(core.field_gradient(moving, events))            # [rows, columns] Antivector-
        yield events, field, current


def circling(orbit: core.Orbit) -> Iterator[tuple[float, core.Antivector]]:
    """Over one turn: each lab time, and the charge then; the pictures ask for the field at each of their samples."""
    period = 2 * np.pi * orbit.radius / orbit.speed
    for time in period * np.arange(n_orbit_frames) / n_orbit_frames:
        position, _, _ = core.worldline(orbit, core.mv.scalar([time]))       # [1] Antivector-
        yield time, position


# --- plumbing -------------------------------------------------------------------------
def pinned() -> None:
    """The picture the repository tracks: the fast charge's waves over one turn, for the docs' gallery."""
    from examples.util.animation import pin_animation
    from examples.electromagnetism.moving_charge import render

    orbit = core.Orbit(orbit_charge, orbit_radius, orbit_speeds[-1])
    frames = render.animate_waves(core.orbit_field_source, orbit, circling(orbit), orbit_half_width, n_orbit_pixels)
    pin_animation(frames, "radiating_charge_fast", duration_ms, colors=n_orbit_colours)


def main() -> None:
    from examples.util.animation import save_animation
    from examples.electromagnetism.moving_charge import render

    # Slowing down plays the rise back.
    save_animation(render.animate(speed_up()), "moving_charge", duration_ms, forward_then_back=True)
    for name, speed in zip(orbit_names, orbit_speeds):
        orbit = core.Orbit(orbit_charge, orbit_radius, speed)
        frames = render.animate_waves(core.orbit_field_source, orbit, circling(orbit), orbit_half_width, n_orbit_pixels)
        save_animation(frames, f"radiating_charge_{name}", duration_ms, colors=n_orbit_colours)

    # checks
    # At the top speed the field's derivative vanishes, and the current is the softened density carried
    # along the charge's own time.
    moving = charge(top_rapidity)
    events = core.grid(half_width, half_height, n_check_columns, n_check_rows)  # [rows, columns] Antivector-
    gradients = core.field_gradient(moving, events)
    np.testing.assert_allclose((core.Vector ^ gradients(core.Antivector)).trace(1, 2).kernel, 0.0, atol=1e-12)
    at_rest = moving.boost << events
    separations = at_rest - (core.mv.t & at_rest) * core.time
    softened = core_radius**2 + (separations | separations)
    density = 3 * strength * core_radius**2 / (4 * np.pi) / (softened * softened * softened.square_root())
    np.testing.assert_allclose((core.current(gradients) - density * (moving.boost >> core.time)).kernel, 0.0, atol=1e-9)


if __name__ == "__main__":
    main()
