"""Foucault pendulums at four northern latitudes turn across a sidereal day, with the animated Panthéon globe in Paris."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.mechanics.foucault import core

# Equator, thirty degrees north, the Panthéon in Paris, and the north pole.
n_latitudes = 4
latitudes = np.radians(np.array([0.0, 30.0, 48.85, 90.0]))                    # [n_latitudes]
paris = 2
# Parallel-transport step count around each latitude circle, animation frame count, and frame duration in milliseconds.
n_steps = 720
n_frames = 144
frame_duration_ms = 40


def paris_day(n_frames: int) -> tuple[core.Point, core.Point, core.Point, Iterator[tuple[core.Rotor, core.Point]]]:
    """Pairs the local `(vertical, north, east)` (`[] Point-`) at the Panthéon in Paris with the `n_frames + 1` daily state stream."""
    vertical, north, east = core.compass(np.asarray(latitudes[paris]))        # [] Point-, Point-, Point-
    return vertical, north, east, core.day_states(float(latitudes[paris]), n_frames)


def main() -> None:
    from examples.util.animation import save_animation
    from examples.mechanics.foucault import render

    vertical, north_paris, east_paris, states = paris_day(n_frames)
    recorded = []

    def recording() -> Iterator[tuple[core.Rotor, core.Point]]:
        for state in states:
            recorded.append(state)
            yield state

    save_animation(render.day_animation(vertical, north_paris, east_paris, recording()), "foucault_day", frame_duration_ms)

    # checks
    _, north, east = core.compass(latitudes)                                  # [n_latitudes] Point-, Point-, Point-
    floor = east.commutator(north)                                            # [n_latitudes] Bivector+
    expected = (floor * (-np.pi * np.sin(latitudes))).exp() >> north          # [n_latitudes] Point-
    full_day = np.full(n_latitudes, 2 * np.pi)                                # [n_latitudes]
    day_ends = core.swing_direction(latitudes, full_day)                      # [n_latitudes] Point-
    assert ((day_ends - expected).antinorm() < 1e-10).all()
    transported = core.parallel_holonomy(latitudes, n_steps) >> north         # [n_latitudes] Point-
    assert ((transported - expected).antinorm() < 1e-4).all()
    final_earth, final_swing = recorded[-1]                                   # [] Rotor+, [] Point-
    assert (final_earth >> north_paris - north_paris).antinorm() < 1e-10
    assert (final_swing - expected[paris]).antinorm() < 1e-10


if __name__ == "__main__":
    main()
