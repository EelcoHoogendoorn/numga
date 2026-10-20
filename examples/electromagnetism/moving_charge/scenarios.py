"""Moving charges: a soft charge sped up to near the speed of light and slowed down again, and a point
charge circling slowly and fast, shedding waves."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.electromagnetism.moving_charge import core

CHARGE = 1.0
CORE_RADIUS = 0.08
TOP_RAPIDITY = 2.0
HALF_WIDTH = 1.5
HALF_HEIGHT = 1.0
N_COLUMNS = 300
N_ROWS = 200
N_FRAMES = 90
DURATION_MS = 60

ORBIT_CHARGE = 1.0
ORBIT_RADIUS = 1.0
# A slow and a fast orbit, as fractions of the speed of light, with the names of their animations.
ORBIT_SPEEDS = np.array([0.5, 0.9])
ORBIT_NAMES = ("slow", "fast")
ORBIT_HALF_WIDTH = 10.0
N_ORBIT_PIXELS = 300
N_ORBIT_FRAMES = 60
# The coarser window the checks sample.
N_CHECK_COLUMNS = 30
N_CHECK_ROWS = 20


# --- math -----------------------------------------------------------------------------
def charge(rapidity: float) -> core.Charge:
    """The charge moving along x at the given rapidity, through the origin at lab time zero."""
    return core.Charge(np.array(CHARGE), CORE_RADIUS, (core.mv.xt * (rapidity / 2)).exp(), 0 * core.time)


def speed_up() -> Iterator[tuple[core.Antivector, core.Bivector, core.Antivector]]:
    """The lab's view at time zero as the charge speeds up: the events, the field, and the current."""
    events = core.grid(HALF_WIDTH, HALF_HEIGHT, N_COLUMNS, N_ROWS)               # [rows, columns] Antivector
    # Slowing down retraces speeding up: the rise alone is half the frames.
    rapidities = TOP_RAPIDITY * (1 - np.cos(2 * np.pi * np.arange(N_FRAMES // 2 + 1) / N_FRAMES)) / 2
    for rapidity in rapidities:
        moving = charge(rapidity)
        field = core.field(core.potential_gradient(moving, events))           # [rows, columns] Bivector
        current = core.current(core.field_gradient(moving, events))            # [rows, columns] Antivector
        yield events, field, current


def circling(orbit: core.Orbit) -> Iterator[tuple[core.Antivector, core.Bivector, core.Antivector]]:
    """Over one turn: the events in the orbit's plane at each lab time, the field there, and the
    charge."""
    plane = core.grid(ORBIT_HALF_WIDTH, ORBIT_HALF_WIDTH, N_ORBIT_PIXELS, N_ORBIT_PIXELS)   # [rows, columns] Antivector
    period = 2 * np.pi * orbit.radius / orbit.speed
    for time in period * np.arange(N_ORBIT_FRAMES) / N_ORBIT_FRAMES:
        events = plane + core.time * time                                    # [rows, columns] Antivector
        field = core.field(core.orbit_potential_gradient(orbit, events))     # [rows, columns] Bivector
        position, _, _ = core.worldline(orbit, core.mv.scalar([time]))       # [1] Antivector
        yield events, field, position


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation
    from examples.electromagnetism.moving_charge import render

    speeding_up = render.animate(speed_up())
    # Slowing down plays the rise back.
    save_animation(speeding_up + speeding_up[-2:0:-1], "moving_charge", DURATION_MS)
    for name, speed in zip(ORBIT_NAMES, ORBIT_SPEEDS):
        orbit = core.Orbit(ORBIT_CHARGE, ORBIT_RADIUS, speed)
        save_animation(render.animate_waves(list(circling(orbit))), f"radiating_charge_{name}", DURATION_MS)

    # checks
    # At the top speed the field's derivative vanishes, and the current is the softened density carried
    # along the charge's own time.
    moving = charge(TOP_RAPIDITY)
    events = core.grid(HALF_WIDTH, HALF_HEIGHT, N_CHECK_COLUMNS, N_CHECK_ROWS)  # [rows, columns] Antivector
    gradients = core.field_gradient(moving, events)
    np.testing.assert_allclose((core.Vector ^ gradients(core.Antivector)).trace(1, 2).kernel, 0.0, atol=1e-12)
    at_rest = moving.boost << events
    separations = at_rest - (core.mv.t & at_rest) * core.time
    softened = CORE_RADIUS**2 - (separations | separations)
    density = 3 * CHARGE * CORE_RADIUS**2 / (4 * np.pi) / (softened * softened * softened.square_root())
    np.testing.assert_allclose((core.current(gradients) - density * (moving.boost >> core.time)).kernel, 0.0, atol=1e-9)


if __name__ == "__main__":
    main()
