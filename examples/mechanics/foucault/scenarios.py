"""Foucault pendulums at four latitudes through a day, and the one in Paris with the earth turning under it."""

from __future__ import annotations

import numpy as np

from examples.mechanics.foucault import core

# The latitudes: the equator, thirty degrees north, the Panthéon in Paris where Foucault hung his pendulum,
# and the north pole.
LATITUDES = np.radians(np.array([0.0, 30.0, 48.85, 90.0]))
PARIS = 2
# The steps a direction is carried around a latitude circle in.
N_STEPS = 720
# The frames of the day in Paris, and the duration of each.
N_FRAMES = 72
DURATION_MS = 80


# --- math -----------------------------------------------------------------------------
def day() -> tuple[core.Rotor, core.Point, core.Point, core.Point, core.Point]:
    """For the pendulum in Paris at every frame of a day: the rotor of the earth's turn so far, and the line
    the pendulum swings along on its floor; with the site's vertical, north and east at the start of the day."""
    turns = np.linspace(0, 2 * np.pi, N_FRAMES + 1)                              # [frames + 1]
    vertical, north, east = core.compass(LATITUDES[PARIS])
    return core.turn(core.equator, turns), core.swing(LATITUDES[PARIS], turns), vertical, north, east


# --- plumbing -------------------------------------------------------------------------
def turned(directions: core.Point, north: core.Point, east: core.Point) -> np.ndarray:
    """How far each direction on its floor has turned from north, clockwise seen from above."""
    return np.arctan2(directions.dot(east).to_array(), directions.dot(north).to_array())


def main() -> None:
    from examples.util.animation import save_animation
    from examples.mechanics.foucault import render

    save_animation(render.animate_day(*day()), "foucault_day", DURATION_MS)

    # checks
    _, north, east = core.compass(LATITUDES)
    # After a day each swing has turned clockwise from north by the sine of the latitude times a full turn,
    # and a direction carried once around the latitude circle comes back turned the same way.
    ends = core.swing(LATITUDES, np.full(len(LATITUDES), 2 * np.pi))         # [latitudes] Point
    expected = 2 * np.pi * np.sin(LATITUDES)
    assert np.abs(np.sin(turned(ends, north, east) - expected)).max() < 1e-10
    carried = core.carried(LATITUDES, N_STEPS) >> north                          # [latitudes] Point
    assert np.abs(np.sin(turned(carried, north, east) - expected)).max() < 1e-4


if __name__ == "__main__":
    main()
