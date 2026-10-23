"""Sundials: a horizontal dial at a northern latitude, with its hour lines, the shadow paths of the seasons,
and the sun's cone of rays turning through a year."""

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
# is drawn through, from solstice to solstice; and the samples of each day's path.
N_MONTHS = 3
HOURS = np.arange(6, 19)
N_DECLINATIONS = 25
DAY_HOURS = np.arange(3, 21.01, 0.05)
# The frames of one year, the hours of a day each frame's rays are drawn at, and the duration of a frame.
N_FRAMES = 48
RAY_HOURS = np.arange(4, 21, 0.25)
DURATION_MS = 90


# --- math -----------------------------------------------------------------------------
def dial() -> tuple[core.Point, np.ndarray, core.Point, core.Point, np.ndarray]:
    """The nodus, the seasons' declinations and the sun through each of their days `[seasons, samples]`,
    and the sun along each hour line from the winter solstice to the summer one `[hours, samples]`, with
    the hours."""
    pole, equator = core.sky(LATITUDE)
    seasons = TILT * np.sin(np.pi / 2 * np.arange(-N_MONTHS, N_MONTHS + 1) / N_MONTHS)   # [seasons]
    days = core.sun(pole, equator, seasons[:, None], hour_angle(DAY_HOURS)[None, :])   # [seasons, samples] Point
    declinations = np.linspace(-TILT, TILT, N_DECLINATIONS)                # [samples]
    hours = core.sun(pole, equator, declinations[None, :], hour_angle(HOURS)[:, None])   # [hours, samples] Point
    return core.up * HEIGHT, seasons, days, hours, HOURS


def year() -> tuple[np.ndarray, core.Point, core.Point]:
    """At every frame of a year: the sun's declination, the sun at each of the day's hours `[frames,
    hours]`, and through the whole day `[frames, samples]`."""
    pole, equator = core.sky(LATITUDE)
    declinations = TILT * np.sin(np.linspace(0, 2 * np.pi, N_FRAMES, endpoint=False))   # [frames]
    rays = core.sun(pole, equator, declinations[:, None], hour_angle(RAY_HOURS)[None, :])   # [frames, hours] Point
    days = core.sun(pole, equator, declinations[:, None], hour_angle(DAY_HOURS)[None, :])   # [frames, samples] Point
    return declinations, rays, days


# --- plumbing -------------------------------------------------------------------------
def hour_angle(hours: np.ndarray) -> np.ndarray:
    """The sun's hour angle at each hour of local solar time: a fifteenth of a quarter turn per hour from
    noon, westwards in the afternoon."""
    return np.radians(15.0 * (hours - 12))


def main() -> None:
    from examples.animation import save_animation, save_figure
    from examples.geometry.sundial import render

    nodus, seasons, days, suns, hours = dial()
    pole, _ = core.sky(LATITUDE)
    foot = core.shadow(nodus, pole)                                             # [] Point
    save_figure(render.draw_dial(nodus, foot, seasons, days, suns, hours), "sundial")
    save_animation(render.animate_year(*year(), nodus, foot, seasons, days, suns, hours), "sundial_year", DURATION_MS)

    # checks
    # Every shadow of the nodus through each season's day lies on that season's cone of rays.
    rays = (nodus - core.shadow(nodus, days)).normalized()                      # [seasons, samples] Point
    assert np.abs((core.cone(pole, days[:, 0])[:, None](rays) & rays).to_array()).max() < 1e-10
    # Each hour's shadows at every season lie on one line through the foot of the style, at the angle
    # of gnomonics.
    tips = core.shadow(nodus, suns)                                             # [hours, samples] Point
    angles = render.bearings(tips - foot)                                       # [hours, samples]
    expected = np.arctan2(np.sin(LATITUDE) * np.sin(hour_angle(hours)), np.cos(hour_angle(hours)))
    assert np.abs(np.sin(angles - expected[:, None])).max() < 1e-10


if __name__ == "__main__":
    main()
