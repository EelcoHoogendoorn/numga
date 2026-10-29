"""Scenes for a horizontal sundial at a northern latitude compute its hour lines, the seasonal shadow paths of the nodus, and the daily cone of rays sweeping between the solstices."""

from __future__ import annotations

import numpy as np

from examples.geometry.sundial import core

# The dial's latitude, the height of the nodus above the dial, and the tilt of the earth's axis, the
# declination of the sun at the solstices.
LATITUDE = np.radians(52.0)
HEIGHT = 1.0
TILT = np.radians(23.44)
# The seasons drawn: the sun's declination a whole number of months from an equinox, up to a solstice on
# either side; the hours drawn, from six in the morning to six at night; the declinations each hour line
# is drawn through, from solstice to solstice; and the samples of each day's path, with their hour angles
# from solar noon at a twenty-fourth of a turn per hour.
NOON = 12
TURN_PER_HOUR = 2 * np.pi / 24
N_MONTHS = 3
HOURS = np.arange(6, 19)
HOUR_ANGLES = (HOURS - NOON) * TURN_PER_HOUR
N_DECLINATIONS = 25
DAY_HOURS = np.arange(3, 21.01, 0.05)
DAY_ANGLES = (DAY_HOURS - NOON) * TURN_PER_HOUR
# The frames of one year, the hours of a day each frame's rays are drawn at, and the duration of a frame.
N_FRAMES = 108
RAY_HOURS = np.arange(4, 21, 0.25)
RAY_ANGLES = (RAY_HOURS - NOON) * TURN_PER_HOUR
DURATION_MS = 40


# --- math -----------------------------------------------------------------------------
def dial() -> tuple[core.Point, np.ndarray, core.Point, core.Point, np.ndarray]:
    """The dial's scene is the nodus, the seasons' declinations and daily sun paths, and the sun directions along each hour line from solstice to solstice."""
    pole, equator = core.sky(LATITUDE)                                                  # [] Point-, Point-
    seasons = TILT * np.sin(np.pi / 2 * np.arange(-N_MONTHS, N_MONTHS + 1) / N_MONTHS)  # [seasons]
    days = core.sun(pole, equator, seasons[:, None], DAY_ANGLES[None, :])               # [seasons, samples] Point-
    declinations = np.linspace(-TILT, TILT, N_DECLINATIONS)                             # [declinations]
    hour_suns = core.sun(pole, equator, declinations[None, :], HOUR_ANGLES[:, None])    # [hours, declinations] Point-
    return core.up * HEIGHT, seasons, days, hour_suns, HOURS


def year() -> tuple[np.ndarray, core.Point, core.Point]:
    """The year's frames are the sun's declination, its rays at the drawn hours and its daily path, from winter solstice to summer solstice."""
    pole, equator = core.sky(LATITUDE)                                                   # [] Point-, Point-
    declinations = TILT * np.sin(np.linspace(-np.pi / 2, np.pi / 2, N_FRAMES // 2 + 1))  # [frames]
    rays = core.sun(pole, equator, declinations[:, None], RAY_ANGLES[None, :])           # [frames, ray_hours] Point-
    days = core.sun(pole, equator, declinations[:, None], DAY_ANGLES[None, :])           # [frames, samples] Point-
    return declinations, rays, days


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.geometry.sundial import render

    nodus, seasons, days, suns, hours = dial()
    pole, _ = core.sky(LATITUDE)                                               # [] Point-, Point-
    foot = core.shadow(nodus, pole)                                            # [] Point-
    save_figure(render.draw_dial(nodus, foot, seasons, days, suns, hours), "sundial")
    save_animation(render.animate_year(*year(), nodus, foot, seasons, days, suns, hours), "sundial_year", DURATION_MS,
                   forward_then_back=True)

    # checks
    # Every shadow of the nodus through each season's day lies on that season's cone of rays.
    rays = (nodus - core.shadow(nodus, days)).normalized()                     # [seasons, samples] Point-
    assert np.abs((core.cone(pole, days[:, 0])[:, None](rays) & rays).to_array()).max() < 1e-10
    # Each hour's shadows at every season lie on one line through the foot of the style, at the angle
    # of gnomonics.
    offsets = core.shadow(nodus, suns) - foot                                  # [hours, declinations] Point-
    angles = np.arctan2((core.mv.x & offsets).to_array(), (core.mv.y & offsets).to_array())
    expected = np.arctan2(np.sin(LATITUDE) * np.sin(HOUR_ANGLES), np.cos(HOUR_ANGLES))
    assert np.abs(np.sin(angles - expected[:, None])).max() < 1e-10


if __name__ == "__main__":
    main()
