"""Rendering for the horizontal sundial draws the dial face with its hour lines and seasonal conics as strokes read by the dial's planes east and north, with matplotlib's numerals over them, and traces the sun's daily cone of rays past the nodus in space on the GPU."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np

from examples.util import charts
from examples.util.animation import capture
from examples.geometry.sundial import core
from numga import concatenate, stack
from examples.util.wgsl import fitted, show, simplices, solids
from examples.util.wgsl.camera import Camera, flat_panel

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
# The widths on the dial's face, in pixels: the hour lines a frame's width, the seasons' paths a line's, the day's path
# twice that and the gnomon's base three times; and how much the seasons' paths cover behind the day's.
hour_width, season_width = charts.FRAME_PX, charts.LINE_PX
day_width, base_width = 2 * charts.LINE_PX, 3 * charts.LINE_PX
faint_opacity = 0.35
plate_thickness, gnomon_thickness = 0.25, 0.08
# The dial's corners, east, north and up.
dial_corners = solids.mv.point([[east_range[0], north_range[0], 0], [east_range[1], north_range[0], 0],
                                [east_range[1], north_range[1], 0], [east_range[0], north_range[1], 0]])   # [corners] Point-
# The planes across the dial's face, east and north, that read where a shadow falls on it.
face_planes = stack((core.mv.x, core.mv.y))                                    # [2] Plane+


# --- plumbing -------------------------------------------------------------------------
def on_face(nodus: core.Point, suns: core.Point) -> np.ndarray:
    """Whether the sun is up and the nodus's shadow falls on the dial face `[...]`."""
    tips = core.shadow(nodus, suns)                                            # [...] Point-
    east, north = core.mv.x & tips, core.mv.y & tips                           # [...] Scalar+, Scalar+
    return ((core.dial & suns) > 0) & (east > east_range[0]) & (east < east_range[1]) \
        & (north > north_range[0]) & (north < north_range[1])


def sunlit_chains(suns: core.Point) -> simplices.Topology:
    """The shadow paths of a batch of suns as chains of edges along its last axis, an edge kept where the sun is up at
    both its ends; the face's camera cuts them at the dial's edges."""
    n_samples = suns.shape[-1]
    chains = simplices.Topology.chains([n_samples] * (np.prod(suns.shape) // n_samples))
    # The dial's reading of the sun at both ends of each edge, positive where it is up.
    readings = core.dial & (chains.incidence[1, 0] * suns.reshape(-1).field())   # [ends] Scalar+[edges]
    return chains.subset((readings > 0).all(axis=0), 1)


def shadow_strokes(nodus: core.Point, suns: core.Point, colours: np.ndarray, width: float, camera: Camera,
                   opacity: float = 1.0) -> np.ndarray:
    """The shadow paths of the nodus under a batch of suns, each run along the last axis, as strokes on the dial, each
    edge in its colour `[edges, 3]`."""
    return simplices.strokes(core.shadow(nodus, suns).reshape(-1), sunlit_chains(suns), colours, width, camera,
                             opacity=opacity)


def season_colours(seasons: np.ndarray) -> np.ndarray:
    """Colour each season by its distance from the equinox so that conjugate summer and winter declinations share a colour."""
    return plt.get_cmap(season_map)(0.15 + 0.7 * np.abs(seasons) / np.abs(seasons).max())


def draw_face(ax: plt.Axes, nodus: core.Point, foot: core.Point, seasons: np.ndarray, days: core.Point,
              suns: core.Point, hours: np.ndarray, faint: bool) -> tuple[Camera, list[np.ndarray]]:
    """Draw the dial face with each season's shadow path, the numbered hour lines from solstice to solstice, and the
    gnomon's base along the noon line: the face's camera and its pictures, the numerals written over them."""
    ax.set(xlim=east_range, ylim=north_range, aspect="equal")
    ax.set_axis_off()
    camera = flat_panel(ax, core.ga)
    # Each season's path in its colour, one per edge of its run.
    season_edges = sunlit_chains(days).elements[1][:, 0] // days.shape[-1]    # [edges]
    paths = shadow_strokes(nodus, days, season_colours(seasons)[season_edges, :3], season_width, camera,
                           faint_opacity if faint else 1.0)
    # The hour lines thin, a frame's width, under the seasons' paths.
    lines = shadow_strokes(nodus, suns, np.array(to_rgb(hour_colour)), hour_width, camera)
    hour_lines = core.shadow(nodus, suns)                                      # [hours, declinations] Point-
    for hour, line, drawn in zip(hours, hour_lines, on_face(nodus, suns)):
        # The numeral just inside the outer end of the line's part on the dial.
        outer, inner = line[np.flatnonzero(drawn)[[0, -1]]]                    # [] Point-, Point-
        place = outer - inset * (outer - inner).normalized()                   # [] Point-
        ax.text(*(face_planes & place[None]).to_array(), f"{hour}", color=hour_colour, ha="center", va="center",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
    # The gnomon's edge seen from above, bold and square at its ends: from the foot of the style to the point of the
    # dial below the nodus.
    below_nodus = nodus - core.up * (core.dial & nodus)                        # [] Point-
    base = simplices.strokes(stack([foot, below_nodus]), simplices.Topology.chains([2]), np.array(to_rgb(gnomon_colour)),
                             base_width, camera, cap=simplices.Cap.BUTT)
    return camera, [lines, paths, base]


@charts.style()
def draw_dial(nodus: core.Point, foot: core.Point, seasons: np.ndarray, days: core.Point, suns: core.Point,
              hours: np.ndarray) -> plt.Figure:
    """Render the static dial face figure."""
    # Over half the column: the hour lines, their numerals and the seasons' paths across the dial.
    figure, ax = charts.subplots(460, 345, layout="constrained")
    _, pictures = draw_face(ax, nodus, foot, seasons, days, suns, hours, faint=False)
    simplices.overlay(ax, pictures)
    return figure


def daylight(nodus: core.Point, rays: core.Point) -> tuple[core.Point, core.Point]:
    """The daytime rays' starts, sunward of the nodus, and their shadows on the dial plane."""
    daytime = (core.dial & rays) > 0                                           # [ray_hours]
    return nodus + sunward * rays[daytime], core.shadow(nodus, rays[daytime])   # [daytime] Point-, Point-


def sky_view(ax: plt.Axes, nodus: core.Point, rays: core.Point) -> solids.View:
    """Frame a view of the dial plate and all daytime rays across the year to fit the axes."""
    sunward_points = concatenate([daylight(nodus, day)[0] for day in rays])    # [rays] Point-
    return fitted(solids.view([dial_corners, solids.origin + solids.context.include_from_algebra(sunward_points)], 0.0, elevation, azimuth,
                              margin, 1), ax)


def day_path(nodus: core.Point, foot: core.Point, day: core.Point) -> solids.Ink:
    """The ink of a day's shadow path, where the dial plate cuts the cone of the day's rays about the pole, the cone
    carried from the origin to its apex at the nodus."""
    # The style runs from its foot to the nodus, towards the pole.
    pole = (nodus - foot).normalized()                                         # [] Point-
    to_nodus = solids.translator(solids.context.include_from_algebra(nodus))                    # [] Motor+
    # The cone of space about the origin reads each point of the renderer's space by its offset from the origin, its
    # part along the degenerate blades.
    about_origin = solids.context.include_from_algebra(core.cone(pole, day[0]))(solids.Point.restrict.degenerate())   # [] Plane+ <- Point-
    # The shadow lies away from the sun, on the negative side of the plane through the nodus across the pole, weighted
    # by the sun's height along the pole, the sine of its declination.
    across = pole.dual_inverse()                                               # [] Plane+
    shadow_side = solids.half_space(to_nodus >> solids.context.include_from_algebra(across * (across & day[0])))
    return solids.Ink(0, solids.placed(about_origin, to_nodus)[None], shadow_side[None], np.array(to_rgb(day_colour)),
                      path_width, 1.0)


def dial_in_space(nodus: core.Point, foot: core.Point) -> list[tuple[solids.Quadric, np.ndarray]]:
    """Build the dial plate and the triangular gnomon as solids in space."""
    # The plate's box, its top face the dial's, carried from the origin to its middle.
    middle = dial_corners.mean(axis=0) - solids.mv.z.dual() * (plate_thickness / 2)  # [] Point-
    half_extents = np.array([np.ptp(east_range) / 2, np.ptp(north_range) / 2, plate_thickness / 2])
    slab = solids.boxes(solids.translator(middle - solids.origin), half_extents)[None]  # [1, 6] Plane+
    # The gnomon's triangle joins the foot of the style, the nodus, and the point of the dial below the nodus.
    below_nodus = nodus - core.up * (core.dial & nodus)                        # [] Point-
    corners = solids.origin + solids.context.include_from_algebra(stack([foot, nodus, below_nodus]))       # [corners] Point-
    gnomon = solids.plates(corners[None], gnomon_thickness)                    # [1, 5] Plane+
    return [(slab, np.array(to_rgb(dial_colour))), (gnomon, np.array(to_rgb(gnomon_colour)))]


def sunlight(nodus: core.Point, foot: core.Point, rays: core.Point, day: core.Point,
             view: solids.View) -> tuple[list[tuple[solids.Quadric, np.ndarray]], list[solids.Ink]]:
    """Build the daytime sun rays clipped to the walls above the dial plate and the inked shadow conic on the dial."""
    ray_starts, shadow_points = daylight(nodus, rays)                           # [daytime] Point-, Point-
    # Each ray is clipped where it leaves the space above the dial, at the walls standing on its edges, so that a ray
    # whose shadow moves off the dial shortens at its edge rather than vanishing.
    sides = stack([-solids.mv.x, solids.mv.x, -solids.mv.y, solids.mv.y])      # [4] Plane+
    corners = dial_corners[[0, 2, 0, 2]]                                       # [4] Point-
    walls = solids.half_space(sides - solids.mv.w * (sides & corners))
    rods = solids.rods(solids.origin + solids.context.include_from_algebra(ray_starts), solids.origin + solids.context.include_from_algebra(shadow_points),
                       ray_width / 2 * view.pixel)                             # [rays, 3] Plane+ <- Point-
    beams = concatenate([rods, walls[None].broadcast_to((rods.shape[0], 4))], axis=-1)                # [rays, 7] Plane+ <- Point-
    # Blend the ray colour over the white background according to its opacity.
    beam_colour = ray_opacity * np.array(to_rgb(ray_colour)) + (1 - ray_opacity)
    return [(beams, beam_colour)], [day_path(nodus, foot, day)]


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
        camera, pictures = draw_face(face, nodus, foot, seasons, season_days, suns, hours, faint=True)
    # The day's shadow path, bold, laid over the face each frame.
    path = simplices.overlay(face, pictures)
    view = sky_view(sky_axes, nodus, rays)
    sky = show(sky_axes, np.ones((*view.shape, 3)), view)
    for index in range(len(declinations)):
        beams, inks = sunlight(nodus, foot, rays[index], days[index], view)
        sky.set_data(solids.draw(dial + beams, view, np.ones(3), inks, translucent=[], meshes=[]))
        day = shadow_strokes(nodus, days[index], np.array(to_rgb(day_colour)), day_width, camera)
        path.set_data(simplices.composited(pictures + [day]))
        yield capture(figure)
    plt.close(figure)
