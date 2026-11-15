"""Scenes for a horizontal sundial at a northern latitude compute its hour lines, the seasonal shadow paths of the nodus, and the daily cone of rays sweeping between the solstices."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.geometry.sundial import core

# Dial latitude, nodus height above the dial, and axial tilt (solstice declination).
latitude = np.radians(52.0)
height = 1.0
tilt = np.radians(23.44)
# Seasonal declinations, drawn hours, solstice-to-solstice samples along each hour line, and daily path samples.
noon = 12
turn_per_hour = 2 * np.pi / 24
n_months = 3
n_seasons = 2 * n_months + 1
hours = np.arange(6, 19)
n_hours = len(hours)
hour_angles = (hours - noon) * turn_per_hour
n_declinations = 25
day_hours = np.arange(3, 21.01, 0.05)
n_samples = len(day_hours)
day_angles = (day_hours - noon) * turn_per_hour
# Half-year sweep frames, drawn ray hours, and frame duration.
n_frames = 108
n_half_year = n_frames // 2 + 1
ray_hours = np.arange(4, 21, 0.25)
n_ray_hours = len(ray_hours)
ray_angles = (ray_hours - noon) * turn_per_hour
frame_duration_ms = 40


# --- math -----------------------------------------------------------------------------
def dial_scene() -> tuple[core.Point, np.ndarray, core.Point, core.Point, np.ndarray]:
    """The dial scene bundles the nodus, seasonal declinations, daily sun paths, hour-line sun directions, and hour labels."""
    pole, equator = core.sky(latitude)                                                   # [] Point-, Point-
    seasons = tilt * np.sin(np.pi / 2 * np.arange(-n_months, n_months + 1) / n_months)   # [n_seasons]
    days = core.sun(pole, equator, seasons[:, None], day_angles[None, :])                # [n_seasons, n_samples] Point-
    declinations = np.linspace(-tilt, tilt, n_declinations)                              # [n_declinations]
    hour_suns = core.sun(pole, equator, declinations[None, :], hour_angles[:, None])     # [n_hours, n_declinations] Point-
    return core.up * height, seasons, days, hour_suns, hours


def annual_envelope() -> core.Point:
    """Daytime ray directions across the half-year declinations (`[n_half_year, n_ray_hours] Point-`) bound the sky view."""
    pole, equator = core.sky(latitude)                                                   # [] Point-, Point-
    declinations = tilt * np.sin(np.linspace(-np.pi / 2, np.pi / 2, n_half_year))        # [n_half_year]
    return core.sun(pole, equator, declinations[:, None], ray_angles[None, :])           # [n_half_year, n_ray_hours] Point-


def annual_states() -> Iterator[tuple[core.Point, core.Point]]:
    """Yields `(day_rays, day_suns)` (`[n_ray_hours] Point-`, `[n_samples] Point-`) from winter solstice to summer solstice."""
    pole, equator = core.sky(latitude)                                                   # [] Point-, Point-
    declinations = tilt * np.sin(np.linspace(-np.pi / 2, np.pi / 2, n_half_year))        # [n_half_year]
    return core.year_states(pole, equator, declinations, ray_angles, day_angles)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.geometry.sundial import render

    nodus, seasons, days, suns, drawn_hours = dial_scene()
    pole, equator = core.sky(latitude)                                                   # [] Point-, Point-
    del equator
    foot = core.shadow(nodus, pole)                                                      # [] Point-
    save_figure(render.dial_figure(nodus, foot, seasons, days, suns, drawn_hours), "sundial")

    recorded: list[tuple[core.Point, core.Point]] = []

    def streaming() -> Iterator[tuple[core.Point, core.Point]]:
        for state in annual_states():
            recorded.append(state)
            yield state

    save_animation(
        render.year_animation(nodus, foot, seasons, days, suns, drawn_hours, annual_envelope(), streaming()),
        "sundial_year",
        frame_duration_ms,
        forward_then_back=True,
    )

    # checks
    assert len(recorded) == n_half_year
    # Every shadow of the nodus through each season's day lies on that season's cone of rays.
    rays = (nodus - core.shadow(nodus, days)).normalized()                               # [n_seasons, n_samples] Point-
    assert ((core.cone(pole, days[:, 0])[:, None](rays) & rays).abs() < 1e-10).all()
    # Each hour's shadows across all seasons lie on one line through the foot of the style at the gnomonic angle.
    offsets = (core.shadow(nodus, suns) - foot).normalized()                             # [n_hours, n_declinations] Point-
    expected_east = np.sin(latitude) * np.sin(hour_angles)[:, None]
    expected_north = np.cos(hour_angles)[:, None]
    residuals = expected_north * (core.mv.x & offsets) - expected_east * (core.mv.y & offsets)  # [n_hours, n_declinations] Scalar+
    assert (residuals.abs() < 1e-10).all()


if __name__ == "__main__":
    main()
