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
east_range = (-4.5, 4.5)
north_range = (-2.0, 4.5)
# How far inside the edge of the dial the hour numerals sit.
inset = 0.25
hour_colour = "0.45"
season_map = "plasma"
day_colour = "#d06c36"
gnomon_colour = "0.15"
dial_colour = "#faf5ea"
ray_colour = "#e8a33d"
# How strongly the rays' colour shows over the white page.
ray_opacity = 0.6
# How far each ray is drawn beyond the nodus towards the sun.
sunward = 2.0
# The view's elevation and azimuth, in radians, azimuth zero looking along +y, the north, and the margin as a share of
# the larger side.
elevation, azimuth = np.radians(16.0), np.radians(-125.0 + 90.0)
margin = 0.04
# The widths of the rays and of the day's shadow path, in pixels; the thickness of the dial's plate and of the gnomon.
ray_width, path_width = 1.8, 3.1
plate_thickness, gnomon_thickness = 0.25, 0.08
# The dial's corners, east, north and up.
dial_corners = solids.mv.point([[east_range[0], north_range[0], 0], [east_range[1], north_range[0], 0],
                                [east_range[1], north_range[1], 0], [east_range[0], north_range[1], 0]])   # [corners] Point-
# The planes across the dial's face, east and north, that read where a shadow falls on it.
face_planes = stack((core.mv.x, core.mv.y))                                    # [2] Plane+


# --- plumbing -------------------------------------------------------------------------
def gpu_points(points: core.Point) -> solids.Point:
    """The GPU library's own points at the places of the example's: their pairings with the planes x, y and z, east,
    north and up, are the one hand-off from this algebra to the renderer's."""
    return solids.mv.point((stack((core.mv.x, core.mv.y, core.mv.z)) & points[..., None]).to_array())


def shadows(nodus: core.Point, suns: core.Point) -> np.ndarray:
    """The shadows of the nodus on the dial, east and north `[..., 2]`, masked where the sun is below the horizon or
    the shadow falls outside the dial face."""
    tips = core.shadow(nodus, suns)                                            # [...] Point-
    east, north = core.mv.x & tips, core.mv.y & tips                           # [...] Scalar+, Scalar+
    inside = ((core.dial & suns) > 0) & (east > east_range[0]) & (east < east_range[1]) \
        & (north > north_range[0]) & (north < north_range[1])
    return np.where(inside[..., None], (face_planes & tips[..., None]).to_array(), np.nan)


def season_colours(seasons: np.ndarray) -> np.ndarray:
    """Colour each season by its distance from the equinox so that conjugate summer and winter declinations share a colour."""
    return plt.get_cmap(season_map)(0.15 + 0.7 * np.abs(seasons) / np.abs(seasons).max())


def draw_face(ax: plt.Axes, nodus: core.Point, foot: core.Point, seasons: np.ndarray, days: core.Point,
              suns: core.Point, hours: np.ndarray, faint: bool) -> None:
    """Draw the dial face with each season's shadow path, the numbered hour lines from solstice to solstice, and the gnomon's base along the noon line."""
    paths = shadows(nodus, days)                                               # [seasons, samples, 2]
    ax.add_collection(LineCollection(paths, colors=season_colours(seasons), alpha=0.35 if faint else 1.0))
    lines = shadows(nodus, suns)                                               # [hours, declinations, 2]
    # The hour lines thin, a frame's width, under the seasons' paths.
    ax.add_collection(LineCollection(lines, colors=hour_colour, linewidths=charts.FRAME_WIDTH))
    hour_lines = core.shadow(nodus, suns)                                      # [hours, declinations] Point-
    for hour, line, drawn in zip(hours, hour_lines, ~np.isnan(lines[..., 0])):
        # The numeral just inside the outer end of the line's part on the dial.
        outer, inner = line[np.flatnonzero(drawn)[[0, -1]]]                    # [] Point-, Point-
        place = outer - inset * (outer - inner).normalized()                   # [] Point-
        ax.text(*(face_planes & place[None]).to_array(), f"{hour}", color=hour_colour, ha="center", va="center",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
    # The gnomon's edge seen from above: from the foot of the style to the point of the dial below the nodus.
    below_nodus = nodus - core.up * (core.dial & nodus)                        # [] Point-
    base = (face_planes & stack([foot, below_nodus])[:, None]).to_array()     # [ends, 2]
    # The gnomon's base bold, three lines wide.
    ax.plot(*base.T, color=gnomon_colour, linewidth=3 * charts.LINE_WIDTH, solid_capstyle="butt")
    ax.set(xlim=east_range, ylim=north_range, aspect="equal")
    ax.set_axis_off()


@charts.style()
def draw_dial(nodus: core.Point, foot: core.Point, seasons: np.ndarray, days: core.Point, suns: core.Point,
              hours: np.ndarray) -> plt.Figure:
    """Render the static dial face figure."""
    # Over half the column: the hour lines, their numerals and the seasons' paths across the dial.
    figure, ax = charts.subplots(460, 345, layout="constrained")
    draw_face(ax, nodus, foot, seasons, days, suns, hours, faint=False)
    return figure


def daylight(nodus: core.Point, rays: core.Point) -> tuple[core.Point, core.Point]:
    """The daytime rays' starts, sunward of the nodus, and their shadows on the dial plane."""
    daytime = (core.dial & rays) > 0                                           # [ray_hours]
    return nodus + sunward * rays[daytime], core.shadow(nodus, rays[daytime])   # [daytime] Point-, Point-


def sky_view(ax: plt.Axes, nodus: core.Point, rays: core.Point) -> solids.View:
    """Frame a view of the dial plate and all daytime rays across the year to fit the axes."""
    sunward_points = concatenate([daylight(nodus, day)[0] for day in rays])    # [rays] Point-
    return fitted(solids.view([dial_corners, gpu_points(sunward_points)], 0.0, elevation, azimuth, margin, 1), ax)


def day_path(nodus: core.Point, day: core.Point) -> solids.Ink:
    """Build the ink for a day's shadow path by intersecting the dial plate with the cone swept by the day's sun rays about the pole."""
    suns = gpu_points(day)                                                     # [samples] Point-
    apex = gpu_points(nodus)                                                   # [] Point-
    # The day's suns lie on a circle about the pole, so the pole is square to the plane through them; the cone takes
    # the axial plane squared, so its sense does not matter.
    circle = ((suns[0] & suns[1] & suns[2]) * solids.flip).normalized()        # [] Plane+
    equator = circle - solids.mv.w * (circle & solids.origin)                  # [] Plane+
    sun = suns[0] - solids.origin                                              # [] Point-
    # The sine of the declination: the sun's height along the pole over its length, the size of its dual.
    height = (equator & sun) / sun.dual().norm()                               # [] Scalar+
    # The planes through the nodus: square to the pole, and across x, y and z.
    axial = equator - solids.mv.w * (equator & apex)                           # [] Plane+
    across = solids.axes - solids.mv.w * (solids.axes & apex)                  # [3] Plane+
    # Zero on a ray from the nodus at the sun's angle to the pole: its part along the pole squared, less the sine of the
    # declination squared times its length squared.
    cone = axial * (axial & solids.Point) - height.squared() * (across * (across & solids.Point)).sum(axis=0)  # [] Plane+ <- Point-
    # The shadow lies away from the sun: along the pole, opposite to the sun.
    shadow_side = solids.half_space(axial * height)
    return solids.Ink(0, cone[None], shadow_side[None], np.array(to_rgb(day_colour)), path_width, 1.0)


def dial_in_space(nodus: core.Point, foot: core.Point) -> list[tuple[solids.Quadric, np.ndarray]]:
    """Build the dial plate and the triangular gnomon as solids in space."""
    # The plate's box, its top face the dial's, carried from the origin to its middle.
    middle = dial_corners.mean(axis=0) - solids.mv.z.dual() * (plate_thickness / 2)   # [] Point-
    half_extents = np.array([np.ptp(east_range) / 2, np.ptp(north_range) / 2, plate_thickness / 2])
    slab = solids.boxes(solids.translator(middle - solids.origin), half_extents)[None]  # [1, 6] Plane+
    # The gnomon's triangle joins the foot of the style, the nodus, and the point of the dial below the nodus.
    below_nodus = nodus - core.up * (core.dial & nodus)                        # [] Point-
    gnomon = solids.plates(gpu_points(stack([foot, nodus, below_nodus]))[None], gnomon_thickness)  # [1, 5] Plane+
    return [(slab, np.array(to_rgb(dial_colour))), (gnomon, np.array(to_rgb(gnomon_colour)))]


def sunlight(nodus: core.Point, rays: core.Point, day: core.Point,
             view: solids.View) -> tuple[list[tuple[solids.Quadric, np.ndarray]], list[solids.Ink]]:
    """Build the daytime sun rays clipped to the walls above the dial plate and the inked shadow conic on the dial."""
    ray_starts, shadow_points = daylight(nodus, rays)                          # [daytime] Point-, Point-
    # Each ray is clipped where it leaves the space above the dial, at the walls standing on its edges, so that a ray
    # whose shadow moves off the dial shortens at its edge rather than vanishing.
    sides = stack([-solids.mv.x, solids.mv.x, -solids.mv.y, solids.mv.y])      # [4] Plane+
    corners = dial_corners[[0, 2, 0, 2]]                                       # [4] Point-
    walls = solids.half_space(sides - solids.mv.w * (sides & corners))
    rods = solids.rods(gpu_points(ray_starts), gpu_points(shadow_points), ray_width / 2 * view.pixel)  # [rays, 3] Plane+ <- Point-
    beams = concatenate([rods, walls[None].broadcast_to((rods.shape[0], 4))], axis=-1)                # [rays, 7] Plane+ <- Point-
    # Blend the ray colour over the white background according to its opacity.
    beam_colour = ray_opacity * np.array(to_rgb(ray_colour)) + (1 - ray_opacity)
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
        path = face.add_collection(LineCollection([], colors=day_colour, linewidths=2 * charts.LINE_WIDTH))
    view = sky_view(sky_axes, nodus, rays)
    sky = show(sky_axes, np.ones((*view.shape, 3)), view)
    for index in range(len(declinations)):
        beams, inks = sunlight(nodus, rays[index], days[index], view)
        sky.set_data(solids.draw(dial + beams, view, np.ones(3), inks, translucent=[], meshes=[]))
        path.set_segments(shadows(nodus, days[index])[None])
        yield capture(figure)
    plt.close(figure)
