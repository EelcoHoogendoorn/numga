"""Scene parameters and streaming generators drive a soft charge boosted near the speed of light and
a point charge radiating on a circular orbit."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack

from examples.electromagnetism.moving_charge import core

charge_strength = 1.0
core_radius = 0.08
top_rapidity = 2.0
# Upright lab window at time zero across which a horizontal charge's field pancakes into a vertical band.
half_width = 0.45
half_height = 0.9
n_columns = 150
n_rows = 300
n_frames = 136
duration_ms = 40

# Opposite-charge dipole parameters for the notebook scenarios.
dipole_strengths = np.array([1.0, -1.0])
dipole_offsets = np.array([0.35, -0.35])
dipole_directions = np.array([1.0, -1.0])
dipole_rapidities = np.array([0.0, 2.0])
close_half_width = 0.25
close_half_height = 0.5
n_sweep_frames = 90

orbit_charge = 1.0
orbit_radius = 1.0
orbit_speeds = np.array([0.5, 0.9])
orbit_names = ("slow", "fast")
orbit_half_width = 10.0
n_orbit_pixels = 300
n_orbit_frames = 90
n_orbit_colours = 32


# --- math -----------------------------------------------------------------------------
def sweep_rapidities(max_rapidity: float, n_total_frames: int) -> np.ndarray:
    """A half-cosine rapidity cycle rises from rest to `max_rapidity` and returns to rest across `n_total_frames`."""
    return max_rapidity * (1 - np.cos(2 * np.pi * np.arange(n_total_frames) / n_total_frames)) / 2


def boost_states() -> Iterator[tuple[core.Antivector, core.Bivector, core.Antivector]]:
    """Yields the event grid, electromagnetic field, and four-current (`[n_rows, n_columns]` each) as a
    single charge accelerates from rest to `top_rapidity` and decelerates back to rest."""
    events = core.event_grid(half_width, half_height, n_columns, n_rows)         # [n_rows, n_columns] Antivector-
    origin = 0 * core.time                                                       # [] Antivector-
    for rapidity in sweep_rapidities(top_rapidity, n_frames):
        boost = (core.mv.xt * (rapidity / 2)).exp()                              # [] Rotor+
        potentials = core.potential_gradient(
            np.array(charge_strength), core_radius, boost, origin, events
        )                                                                        # [n_rows, n_columns] Vector+ <- Antivector-
        fields = core.field_gradient(
            np.array(charge_strength), core_radius, boost, origin, events
        )                                                                        # [n_rows, n_columns] Bivector+ <- Antivector-
        yield events, core.field(potentials), core.four_current(fields)


def orbit_states(orbit_speed: float) -> Iterator[tuple[float, core.Antivector]]:
    """Yields the lab time and worldline position (`[1] Antivector-`) across `n_orbit_frames` of one turn."""
    period = 2 * np.pi * orbit_radius / orbit_speed
    for lab_time in period * np.arange(n_orbit_frames) / n_orbit_frames:
        position, _, _ = core.worldline(orbit_radius, orbit_speed, core.mv.scalar([lab_time]))  # [1] Antivector-
        yield float(lab_time), position


# --- plumbing -------------------------------------------------------------------------
def pinned() -> None:
    """Writes the repository-tracked animation of the relativistic circling charge."""
    from examples.electromagnetism.moving_charge import render
    from examples.util.animation import pin_animation

    fast_speed = float(orbit_speeds[-1])
    frames = render.wave_frames(
        core.orbit_field_source,
        orbit_charge,
        orbit_radius,
        fast_speed,
        orbit_half_width,
        n_orbit_pixels,
        orbit_states(fast_speed),
    )
    pin_animation(frames, "radiating_charge_fast", duration_ms, colors=n_orbit_colours)


def main() -> None:
    from examples.electromagnetism.moving_charge import render
    from examples.util.animation import save_animation

    boost_history: list[tuple[core.Antivector, core.Bivector, core.Antivector]] = []

    def streamed_boost_states() -> Iterator[tuple[core.Antivector, core.Bivector, core.Antivector]]:
        for state in boost_states():
            boost_history.append(state)
            yield state

    save_animation(render.boost_frames(streamed_boost_states()), "moving_charge", duration_ms)
    for name, speed in zip(orbit_names, orbit_speeds):
        orbit_speed = float(speed)
        frames = render.wave_frames(
            core.orbit_field_source,
            orbit_charge,
            orbit_radius,
            orbit_speed,
            orbit_half_width,
            n_orbit_pixels,
            orbit_states(orbit_speed),
        )
        save_animation(frames, f"radiating_charge_{name}", duration_ms, colors=n_orbit_colours)

    # checks
    events, top_field, top_current = boost_history[n_frames // 2]
    _, rest_field, _ = boost_history[0]
    top_boost = (core.mv.xt * (top_rapidity / 2)).exp()                          # [] Rotor+
    spacetime_axes = stack((core.mv.x, core.mv.y, core.mv.z, core.mv.t))         # [4] Vector+
    # Peak electric strength rises under Lorentz contraction relative to rest, and the four-current matches the
    # Plummer rest density convected along the boosted worldline velocity across all four spacetime components.
    assert ((top_field & core.time).norm().max() > (rest_field & core.time).norm().max()).all()
    rest_offsets = top_boost << events                                           # [n_rows, n_columns] Antivector-
    separations = rest_offsets - (core.mv.t & rest_offsets) * core.time          # [n_rows, n_columns] Antivector-
    squared_distance = core_radius**2 + (separations | separations)              # [n_rows, n_columns] Scalar+
    plummer_density = (
        3 * charge_strength * core_radius**2 / (4 * np.pi) / (squared_distance * squared_distance * squared_distance.square_root())
    )                                                                            # [n_rows, n_columns] Scalar+
    plummer_current = plummer_density * (top_boost >> core.time)                 # [n_rows, n_columns] Antivector-
    assert ((spacetime_axes & (top_current - plummer_current)[..., None]).abs() < 1e-9).all()


if __name__ == "__main__":
    main()
