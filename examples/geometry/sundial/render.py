"""The face of a horizontal sundial, and the sun's rays past the nodus onto it: the dial in space drawn on the GPU,
its face by matplotlib."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
import numpy as np

from examples.util.animation import capture
from examples.geometry.sundial import core
from numga import concatenate, stack
from examples.util.wgsl import fitted, show, solids

# The part of the dial drawn, east and north of the point below the nodus.
EAST = (-4.5, 4.5)
NORTH = (-2.0, 4.5)
# How far inside the edge of the dial the hour numerals sit.
INSET = 0.25
HOUR_COLOUR = "0.45"
SEASON_MAP = "plasma"
DAY_COLOUR = "#d06c36"
GNOMON_COLOUR = "0.15"
DIAL_COLOUR = "#faf5ea"
RAY_COLOUR = "#e8a33d"
# How strongly the rays' colour shows over the white page.
RAY_OPACITY = 0.6
# How far each ray is drawn beyond the nodus towards the sun.
SUNWARD = 2.0
# The view's elevation and azimuth, in radians, azimuth zero looking along +y, the north, and the margin as a share of
# the larger side.
ELEVATION, AZIMUTH = np.radians(16.0), np.radians(-125.0 + 90.0)
MARGIN = 0.04
# The widths of the rays and of the day's shadow path, in pixels; the thickness of the dial's plate and of the gnomon.
RAY_WIDTH, PATH_WIDTH = 2.3, 3.9
PLATE_THICKNESS, GNOMON_THICKNESS = 0.25, 0.08
# The dial's corners, east, north and up.
DIAL = np.array([[EAST[0], NORTH[0], 0], [EAST[1], NORTH[0], 0], [EAST[1], NORTH[1], 0], [EAST[0], NORTH[1], 0]])


# --- plumbing -------------------------------------------------------------------------
def scalars(values: core.Scalar) -> np.ndarray:
    """The values of a batch of scalars, for drawing and checks."""
    return values.cast(core.ga.subspace.scalar()).kernel[..., 0]


def coordinates(points: core.Point) -> np.ndarray:
    """East, north and up of each point, on the last axis."""
    return points.cast(core.ga.subspace("yz zx xy")).kernel


def shadows(nodus: core.Point, suns: core.Point) -> np.ndarray:
    """East and north of the nodus's shadow for each sun, and nothing where the sun is below the horizon
    or the shadow falls off the dial."""
    tips = coordinates(core.shadow(nodus, suns))[..., :2]
    inside = (scalars(core.dial & suns) > 0) & (tips[..., 0] > EAST[0]) & (tips[..., 0] < EAST[1]) \
        & (tips[..., 1] > NORTH[0]) & (tips[..., 1] < NORTH[1])
    return np.where(inside[..., None], tips, np.nan)


def season_colours(seasons: np.ndarray) -> np.ndarray:
    """A colour per season by how far it lies from the equinox: a summer and a winter season as far from it
    alike."""
    return plt.get_cmap(SEASON_MAP)(0.15 + 0.7 * np.abs(seasons) / np.abs(seasons).max())


def draw_face(ax: plt.Axes, nodus: core.Point, foot: core.Point, seasons: np.ndarray, days: core.Point,
              suns: core.Point, hours: np.ndarray, faint: bool) -> None:
    """The dial's face: each season's shadow path, each hour line through the shadows at that hour from
    solstice to solstice, numbered where it leaves the dial, and the gnomon's edge along the noon line."""
    paths = shadows(nodus, days)                                               # [seasons, samples, 2]
    ax.add_collection(LineCollection(paths, colors=season_colours(seasons), linewidths=1.6, alpha=0.35 if faint else 1.0))
    lines = shadows(nodus, suns)                                               # [hours, samples, 2]
    ax.add_collection(LineCollection(lines, colors=HOUR_COLOUR, linewidths=0.8))
    for hour, line in zip(hours, lines):
        drawn = line[~np.isnan(line[:, 0])]
        outward = drawn[0] - drawn[-1]
        place = drawn[0] - INSET * outward / np.linalg.norm(outward)
        ax.text(*place, f"{hour}", fontsize=8, color=HOUR_COLOUR, ha="center", va="center",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
    # The gnomon's edge seen from above: from the foot of the style to the point below the nodus.
    base = np.stack([coordinates(foot), coordinates(nodus) * [1, 1, 0]])[:, :2]
    ax.plot(*base.T, color=GNOMON_COLOUR, linewidth=3, solid_capstyle="butt")
    ax.set(xlim=EAST, ylim=NORTH, aspect="equal")
    ax.set_axis_off()


def draw_dial(nodus: core.Point, foot: core.Point, seasons: np.ndarray, days: core.Point, suns: core.Point,
              hours: np.ndarray) -> plt.Figure:
    figure, ax = plt.subplots(figsize=(6.0, 4.6), layout="constrained")
    draw_face(ax, nodus, foot, seasons, days, suns, hours, faint=False)
    return figure


def daylight(nodus: core.Point, rays: core.Point) -> tuple[np.ndarray, np.ndarray]:
    """The rays while the sun is up, from beyond the nodus towards the sun, and down to the shadow in the dial's
    plane, east, north and up: `[up, 3]` each."""
    up = scalars(core.dial & rays) > 0                                        # [hours]
    return coordinates(nodus) + SUNWARD * coordinates(rays)[up], coordinates(core.shadow(nodus, rays))[up]


def sky_view(ax: plt.Axes, nodus: core.Point, rays: core.Point) -> solids.View:
    """One view of the dial, its gnomon, and the rays of every day while the sun is up, as many pixels across as the
    axes' room holds."""
    sunward = np.concatenate([daylight(nodus, day)[0] for day in rays])
    return fitted(solids.view([solids.mv.point(DIAL), solids.mv.point(sunward)], 0.0, ELEVATION, AZIMUTH, MARGIN, 1), ax)


def day_path(nodus: core.Point, day: core.Point) -> solids.Ink:
    """The day's shadow path inked on the dial: the cone the day's rays sweep about the pole, through the nodus, on
    the side of it away from the sun. The day's suns lie on a circle about the pole, so the pole is square to the
    plane through them; the cone takes it squared, so its sense does not matter."""
    suns = solids.mv.point(coordinates(day))                                   # [hours] Point
    apex = solids.mv.point(coordinates(nodus))                                 # [] Point
    # The plane through the suns, at unit weight; the equator's, through the origin square to the pole; and its
    # pairing with the first sun's direction at unit length, the sine of the declination.
    circle = ((suns[0] & suns[1] & suns[2]) * solids.flip).normalized()        # [] Plane
    equator = circle - solids.mv.w * (circle & solids.origin)                  # [] Plane
    sun = suns[0] - solids.origin                                              # [] Point
    height = float((equator & sun).to_array() / sun.dual().norm().to_array())
    # The planes through the nodus: square to the pole, and across x, y and z.
    axial = equator - solids.mv.w * (equator & apex)                           # [] Plane
    across = solids.axes - solids.mv.w * (solids.axes & apex)                  # [3] Plane
    # Zero on a ray from the nodus at the sun's angle to the pole: its part along the pole squared, less the sine of the
    # declination squared of its length squared.
    cone = axial * (axial & solids.Point) - height**2 * (across * (across & solids.Point)).sum(axis=0)
    # The shadow lies away from the sun: along the pole, opposite to the sun.
    shadow_side = solids.half_space(axial * height)
    return solids.Ink(0, cone[None], shadow_side[None], np.array(to_rgb(DAY_COLOUR)), PATH_WIDTH, 1.0)


def dial_in_space(nodus: core.Point, foot: core.Point) -> list[tuple[solids.Quadric, np.ndarray]]:
    """The dial's plate in space, the first part, which the day's shadow path is inked on, and its gnomon, as
    solids."""
    middle = np.array([np.mean(EAST), np.mean(NORTH), -PLATE_THICKNESS / 2])
    halves = np.array([np.ptp(EAST) / 2, np.ptp(NORTH) / 2, PLATE_THICKNESS / 2])
    slab = solids.boxes(solids.translator(solids.mv.point(middle) - solids.origin), halves)[None]   # [1, 6] Plane
    tip = solids.mv.point(coordinates(nodus))                                  # [] Point
    # The gnomon's triangle: the foot of the style, the nodus, and the point of the dial below the nodus.
    below = tip - solids.mv.z.dual() * (solids.mv.z & tip)                     # [] Point
    gnomon = solids.plates(stack([solids.mv.point(coordinates(foot)), tip, below])[None], GNOMON_THICKNESS)   # [1, 5] Plane
    return [(slab, np.array(to_rgb(DIAL_COLOUR))), (gnomon, np.array(to_rgb(GNOMON_COLOUR)))]


def sunlight(nodus: core.Point, rays: core.Point, day: core.Point,
             view: solids.View) -> tuple[list[tuple[solids.Quadric, np.ndarray]], list[solids.Ink]]:
    """The day's rays of sunlight past the nodus onto the dial, as solids: the cone they sweep about the pole, on
    both sides of the nodus; and its cut with the dial, the shadow path inked on the dial's plate."""
    sunward, ends = daylight(nodus, rays)
    # Each ray cut where it leaves the space above the dial, at the walls standing on its edges, so that a ray whose
    # shadow moves off the dial shortens at its edge rather than vanishing.
    sides = stack([-solids.mv.x, solids.mv.x, -solids.mv.y, solids.mv.y])     # [4] Plane
    corners = solids.mv.point(DIAL[[0, 2, 0, 2]])                              # [4] Point
    walls = solids.half_space(sides - solids.mv.w * (sides & corners))
    rods = solids.rods(solids.mv.point(sunward), solids.mv.point(ends), RAY_WIDTH / 2 * view.pixel)   # [rays, 3]
    beams = concatenate([rods, walls[None].broadcast_to((len(sunward), 4))], axis=-1)               # [rays, 7]
    # The rays as lightly as they show over the white page.
    beam_colour = RAY_OPACITY * np.array(to_rgb(RAY_COLOUR)) + (1 - RAY_OPACITY)
    return [(beams, beam_colour)], [day_path(nodus, day)]


def animate_year(declinations: np.ndarray, rays: core.Point, days: core.Point, nodus: core.Point, foot: core.Point,
                 seasons: np.ndarray, season_days: core.Point, suns: core.Point, hours: np.ndarray) -> Iterator[np.ndarray]:
    """Per frame of the year: the day's rays past the nodus in space, traced among the dial and its gnomon, and the
    day's shadow path over the dial's face."""
    dial = dial_in_space(nodus, foot)
    # Lay the figure out once; only the sky and the day's shadow path change between frames.
    figure = plt.figure(figsize=(10.0, 4.6))
    sky_axes = figure.add_axes((0.0, 0.0, 0.5, 1.0))
    view = sky_view(sky_axes, nodus, rays)
    sky = show(sky_axes, np.ones((*view.shape, 3)), view)
    face = figure.add_axes((0.52, 0.04, 0.46, 0.92))
    draw_face(face, nodus, foot, seasons, season_days, suns, hours, faint=True)
    path = face.add_collection(LineCollection([], colors=DAY_COLOUR, linewidths=2.2))
    for index in range(len(declinations)):
        beams, inks = sunlight(nodus, rays[index], days[index], view)
        sky.set_data(solids.draw(dial + beams, view, np.ones(3), inks, translucent=[], meshes=[]))
        path.set_segments(shadows(nodus, days[index])[None])
        yield capture(figure)
    plt.close(figure)
