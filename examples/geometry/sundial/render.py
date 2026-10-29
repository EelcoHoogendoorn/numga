"""Rendering for the horizontal sundial draws the dial face with its hour lines and seasonal conics in matplotlib and traces the sun's daily cone of rays past the nodus in space on the GPU."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
import numpy as np

from examples.util import charts
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
RAY_WIDTH, PATH_WIDTH = 1.8, 3.1
PLATE_THICKNESS, GNOMON_THICKNESS = 0.25, 0.08
# The dial's corners, east, north and up.
DIAL = np.array([[EAST[0], NORTH[0], 0], [EAST[1], NORTH[0], 0], [EAST[1], NORTH[1], 0], [EAST[0], NORTH[1], 0]])


# --- plumbing -------------------------------------------------------------------------
def coordinates(points: core.Point) -> np.ndarray:
    """Read out the east, north, and up coordinates of each point on the last axis."""
    return (stack((core.mv.x, core.mv.y, core.mv.z)) & points[..., None]).to_array()


def shadows(nodus: core.Point, suns: core.Point) -> np.ndarray:
    """Project the nodus onto the dial for each sun direction, masking suns below the horizon or shadows outside the dial face."""
    tips = coordinates(nodus - suns * ((core.dial & nodus) / (core.dial & suns)))[..., :2]
    inside = ((core.dial & suns) > 0) & (tips[..., 0] > EAST[0]) & (tips[..., 0] < EAST[1]) \
        & (tips[..., 1] > NORTH[0]) & (tips[..., 1] < NORTH[1])
    return np.where(inside[..., None], tips, np.nan)


def season_colours(seasons: np.ndarray) -> np.ndarray:
    """Colour each season by its distance from the equinox so that conjugate summer and winter declinations share a colour."""
    return plt.get_cmap(SEASON_MAP)(0.15 + 0.7 * np.abs(seasons) / np.abs(seasons).max())


def draw_face(ax: plt.Axes, nodus: core.Point, foot: core.Point, seasons: np.ndarray, days: core.Point,
              suns: core.Point, hours: np.ndarray, faint: bool) -> None:
    """Draw the dial face with each season's shadow path, the numbered hour lines from solstice to solstice, and the gnomon's base along the noon line."""
    paths = shadows(nodus, days)                                               # [seasons, samples, 2]
    ax.add_collection(LineCollection(paths, colors=season_colours(seasons), alpha=0.35 if faint else 1.0))
    lines = shadows(nodus, suns)                                               # [hours, declinations, 2]
    # The hour lines thin, a frame's width, under the seasons' paths.
    ax.add_collection(LineCollection(lines, colors=HOUR_COLOUR, linewidths=charts.FRAME_WIDTH))
    for hour, line in zip(hours, lines):
        segment = line[~np.isnan(line[:, 0])]
        outward = segment[0] - segment[-1]
        place = segment[0] - INSET * outward / np.linalg.norm(outward)
        ax.text(*place, f"{hour}", color=HOUR_COLOUR, ha="center", va="center",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
    # The gnomon's edge seen from above: from the foot of the style to the point below the nodus.
    base = np.stack([coordinates(foot), coordinates(nodus) * [1, 1, 0]])[:, :2]
    # The gnomon's base bold, three lines wide.
    ax.plot(*base.T, color=GNOMON_COLOUR, linewidth=3 * charts.LINE_WIDTH, solid_capstyle="butt")
    ax.set(xlim=EAST, ylim=NORTH, aspect="equal")
    ax.set_axis_off()


@charts.style()
def draw_dial(nodus: core.Point, foot: core.Point, seasons: np.ndarray, days: core.Point, suns: core.Point,
              hours: np.ndarray) -> plt.Figure:
    """Render the static dial face figure."""
    # Over half the column: the hour lines, their numerals and the seasons' paths across the dial.
    figure, ax = charts.subplots(460, 345, layout="constrained")
    draw_face(ax, nodus, foot, seasons, days, suns, hours, faint=False)
    return figure


def daylight(nodus: core.Point, rays: core.Point) -> tuple[np.ndarray, np.ndarray]:
    """Return the ray start points sunward of the nodus and their shadow points on the dial plane for all daytime rays."""
    daytime = (core.dial & rays) > 0                                           # [ray_hours]
    tips = nodus - rays * ((core.dial & nodus) / (core.dial & rays))           # [ray_hours] Point-
    return coordinates(nodus) + SUNWARD * coordinates(rays)[daytime], coordinates(tips)[daytime]


def sky_view(ax: plt.Axes, nodus: core.Point, rays: core.Point) -> solids.View:
    """Frame a view of the dial plate and all daytime rays across the year to fit the axes."""
    sunward_points = np.concatenate([daylight(nodus, day)[0] for day in rays])
    return fitted(solids.view([solids.mv.point(DIAL), solids.mv.point(sunward_points)], 0.0, ELEVATION, AZIMUTH, MARGIN, 1), ax)


def day_path(nodus: core.Point, day: core.Point) -> solids.Ink:
    """Build the ink for a day's shadow path by intersecting the dial plate with the cone swept by the day's sun rays about the pole."""
    suns = solids.mv.point(coordinates(day))                                   # [samples] Point-
    apex = solids.mv.point(coordinates(nodus))                                 # [] Point-
    # The day's suns lie on a circle about the pole, so the pole is square to the plane through them; the cone takes
    # the axial plane squared, so its sense does not matter.
    circle = ((suns[0] & suns[1] & suns[2]) * solids.flip).normalized()        # [] Plane+
    equator = circle - solids.mv.w * (circle & solids.origin)                  # [] Plane+
    sun = suns[0] - solids.origin                                              # [] Point-
    height = float((equator & sun).to_array() / sun.dual().norm().to_array())
    # The planes through the nodus: square to the pole, and across x, y and z.
    axial = equator - solids.mv.w * (equator & apex)                           # [] Plane+
    across = solids.axes - solids.mv.w * (solids.axes & apex)                  # [3] Plane+
    # Zero on a ray from the nodus at the sun's angle to the pole: its part along the pole squared, less the sine of the
    # declination squared times its length squared.
    cone = axial * (axial & solids.Point) - height**2 * (across * (across & solids.Point)).sum(axis=0)  # [] Plane+ <- Point-
    # The shadow lies away from the sun: along the pole, opposite to the sun.
    shadow_side = solids.half_space(axial * height)
    return solids.Ink(0, cone[None], shadow_side[None], np.array(to_rgb(DAY_COLOUR)), PATH_WIDTH, 1.0)


def dial_in_space(nodus: core.Point, foot: core.Point) -> list[tuple[solids.Quadric, np.ndarray]]:
    """Build the dial plate and the triangular gnomon as solids in space."""
    center = np.array([np.mean(EAST), np.mean(NORTH), -PLATE_THICKNESS / 2])
    half_extents = np.array([np.ptp(EAST) / 2, np.ptp(NORTH) / 2, PLATE_THICKNESS / 2])
    slab = solids.boxes(solids.translator(solids.mv.point(center) - solids.origin), half_extents)[None]  # [1, 6] Plane+
    tip = solids.mv.point(coordinates(nodus))                                  # [] Point-
    # The gnomon's triangle joins the foot of the style, the nodus, and the point of the dial below the nodus.
    base_point = tip - solids.mv.z.dual() * (solids.mv.z & tip)                # [] Point-
    gnomon = solids.plates(stack([solids.mv.point(coordinates(foot)), tip, base_point])[None], GNOMON_THICKNESS)  # [1, 5] Plane+
    return [(slab, np.array(to_rgb(DIAL_COLOUR))), (gnomon, np.array(to_rgb(GNOMON_COLOUR)))]


def sunlight(nodus: core.Point, rays: core.Point, day: core.Point,
             view: solids.View) -> tuple[list[tuple[solids.Quadric, np.ndarray]], list[solids.Ink]]:
    """Build the daytime sun rays clipped to the walls above the dial plate and the inked shadow conic on the dial."""
    ray_starts, shadow_points = daylight(nodus, rays)
    # Each ray is clipped where it leaves the space above the dial, at the walls standing on its edges, so that a ray
    # whose shadow moves off the dial shortens at its edge rather than vanishing.
    sides = stack([-solids.mv.x, solids.mv.x, -solids.mv.y, solids.mv.y])      # [4] Plane+
    corners = solids.mv.point(DIAL[[0, 2, 0, 2]])                              # [4] Point-
    walls = solids.half_space(sides - solids.mv.w * (sides & corners))
    rods = solids.rods(solids.mv.point(ray_starts), solids.mv.point(shadow_points), RAY_WIDTH / 2 * view.pixel)  # [rays, 3] Plane+ <- Point-
    beams = concatenate([rods, walls[None].broadcast_to((len(ray_starts), 4))], axis=-1)                         # [rays, 7] Plane+ <- Point-
    # Blend the ray colour over the white background according to its opacity.
    beam_colour = RAY_OPACITY * np.array(to_rgb(RAY_COLOUR)) + (1 - RAY_OPACITY)
    return [(beams, beam_colour)], [day_path(nodus, day)]


def animate_year(declinations: np.ndarray, rays: core.Point, days: core.Point, nodus: core.Point, foot: core.Point,
                 seasons: np.ndarray, season_days: core.Point, suns: core.Point, hours: np.ndarray) -> Iterator[np.ndarray]:
    """Yield each frame of the year with the day's rays past the nodus in space on the left and the day's shadow path across the dial face on the right."""
    dial = dial_in_space(nodus, foot)
    # Lay the figure out once; only the sky and the day's shadow path change between frames. The whole column: the
    # sky's picture beside the dial, whose lines and numerals are made under the charts' style.
    with charts.style():
        figure = plt.figure(figsize=(800 / charts.DPI, 370 / charts.DPI), dpi=charts.DPI)
        sky_axes = figure.add_axes((0.0, 0.0, 0.5, 1.0))
        face = figure.add_axes((0.52, 0.04, 0.46, 0.92))
        draw_face(face, nodus, foot, seasons, season_days, suns, hours, faint=True)
        # The day's shadow path bold, twice a line's width.
        path = face.add_collection(LineCollection([], colors=DAY_COLOUR, linewidths=2 * charts.LINE_WIDTH))
    view = sky_view(sky_axes, nodus, rays)
    sky = show(sky_axes, np.ones((*view.shape, 3)), view)
    for index in range(len(declinations)):
        beams, inks = sunlight(nodus, rays[index], days[index], view)
        sky.set_data(solids.draw(dial + beams, view, np.ones(3), inks, translucent=[], meshes=[]))
        path.set_segments(shadows(nodus, days[index])[None])
        yield capture(figure)
    plt.close(figure)
