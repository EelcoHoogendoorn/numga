"""A soft charge sped up to near the speed of light and slowed down again, and a point charge
circling slowly and fast, shedding waves."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.electromagnetism.moving_charge import core

CHARGE = 1.0
CORE_RADIUS = 0.08
TOP_RAPIDITY = 2.0
HALF_WIDTH = 1.5
HALF_HEIGHT = 1.0
COLUMNS = 300
ROWS = 200
FRAMES = 90
DURATION_MS = 60

ORBIT_CHARGE = 1.0
ORBIT_RADIUS = 1.0
ORBIT_SPEEDS = {"slow": 0.5, "fast": 0.9}
ORBIT_HALF_WIDTH = 10.0
ORBIT_PIXELS = 300
ORBIT_FRAMES = 60


# --- math -----------------------------------------------------------------------------
def charge(rapidity: float) -> core.Charge:
    """The charge moving along x at the given rapidity, through the origin at lab time zero."""
    return core.Charge(np.array(CHARGE), CORE_RADIUS, (core.mv.xt * (rapidity / 2)).exp(), core.mv.vector([0.0, 0.0, 0.0, 0.0]))


def speed_up() -> Iterator[tuple[core.Vector, core.Bivector, core.Vector]]:
    """The lab's view at time zero as the charge speeds up and slows down: the events, the field,
    and the current from the metric-free trace of the field's gradient."""
    events = core.grid(HALF_WIDTH, HALF_HEIGHT, COLUMNS, ROWS)               # [rows, columns] Vector
    rapidities = TOP_RAPIDITY * (1 - np.cos(2 * np.pi * np.arange(FRAMES) / FRAMES)) / 2
    for rapidity in rapidities:
        moving = charge(rapidity)
        field = core.field(core.potential_gradient(moving, events))           # [rows, columns] Bivector
        current, _ = core.sources(core.field_gradient(moving, events))         # [rows, columns] Vector
        yield events, field, current


def circling(orbit: core.Orbit) -> Iterator[tuple[core.Vector, core.Bivector, core.Vector]]:
    """Over one turn: the events in the orbit's plane at each lab time, the field there, and the
    charge."""
    plane = core.grid(ORBIT_HALF_WIDTH, ORBIT_HALF_WIDTH, ORBIT_PIXELS, ORBIT_PIXELS)   # [rows, columns] Vector
    period = 2 * np.pi * orbit.radius / orbit.speed
    for time in period * np.arange(ORBIT_FRAMES) / ORBIT_FRAMES:
        events = plane + core.mv.t * time                                    # [rows, columns] Vector
        field = core.field(core.orbit_potential_gradient(orbit, events))     # [rows, columns] Bivector
        position, _, _ = core.worldline(orbit, core.mv.scalar([time]))       # [1] Vector
        yield events, field, position


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation
    from examples.electromagnetism.moving_charge import render

    states = list(speed_up())
    save_animation(render.animate(states), "moving_charge", DURATION_MS)
    for name, speed in ORBIT_SPEEDS.items():
        orbit = core.Orbit(ORBIT_CHARGE, ORBIT_RADIUS, speed)
        save_animation(render.animate_waves(list(circling(orbit))), f"radiating_charge_{name}", DURATION_MS)

    # --- checks
    # At the top speed: the potential's derivative has no gauge scalar, the field's has no trivector,
    # and the metric-free traces give the current and nothing for magnetic charge.
    moving = charge(TOP_RAPIDITY)
    events = core.grid(HALF_WIDTH, HALF_HEIGHT, 30, 20)                       # [rows, columns] Vector
    potential = core.potential_derivative(core.potential_gradient(moving, events))
    np.testing.assert_allclose((potential - core.field(core.potential_gradient(moving, events))).kernel, 0.0, atol=1e-12)
    gradients = core.field_gradient(moving, events)
    derivative = core.field_derivative(gradients)
    current, magnetic_source = core.sources(gradients)
    np.testing.assert_allclose((derivative - current).kernel, 0.0, atol=1e-9)
    np.testing.assert_allclose(magnetic_source.kernel, 0.0, atol=1e-9)


if __name__ == "__main__":
    main()
