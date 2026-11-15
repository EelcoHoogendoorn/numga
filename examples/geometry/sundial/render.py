"""Rendering for the horizontal sundial draws the dial face with its hour lines and seasonal conics as strokes read by the dial's east and north planes, places matplotlib numerals over them, and traces the sun's daily cone of rays past the nodus in space on the GPU."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np

from examples.util import charts
from examples.util.animation import capture
from examples.geometry.sundial import core
from numga import concatenate, stack
from numga.topology import Topology
from numga.render import Camera, drawing, simplices, solids
from examples.util.panels import fitted, offset_panel, overlay, show

# Part of the dial drawn, east and north of the point below the nodus.
east_range = (-4.5, 4.5)
north_range = (-2.0, 4.5)
# Inset of the hour numerals from the outer end of each hour line on the dial.
inset = 0.25
hour_colour = "0.45"
season_map = "plasma"
day_colour = "#d06c36"
gnomon_colour = "0.15"
dial_colour = "#faf5ea"
ray_colour = "#e8a33d"
# Opacity of the rays blended over the white background.
ray_opacity = 0.6
# Distance each ray extends beyond the nodus towards the sun.
sunward = 2.0
# View elevation and azimuth in radians, with azimuth zero looking along north, and frame margin.
elevation, azimuth = np.radians(16.0), np.radians(-125.0 + 90.0)
margin = 0.04
# Pixel widths of the rays and the day's shadow path in space; thicknesses of the dial plate and gnomon.
ray_width, path_width = 1.8, 3.1
# Pixel widths on the dial face and background opacity of the seasonal paths behind the active day's path.
hour_width, season_width = charts.FRAME_PX, charts.LINE_PX
day_width, base_width = 2 * charts.LINE_PX, 3 * charts.LINE_PX
faint_opacity = 0.35
plate_thickness, gnomon_thickness = 0.25, 0.08
# Dial corners in east, north, and up coordinates.
dial_corners = solids.mv.point([[east_range[0], north_range[0], 0], [east_range[1], north_range[0], 0],
                                [east_range[1], north_range[1], 0], [east_range[0], north_range[1], 0]])   # [4] Point-
# East and north planes across the dial face that read where a shadow falls on it.
face_planes = stack((core.mv.x, core.mv.y))                                    # [2] Plane+


# --- plumbing -------------------------------------------------------------------------
def face_mask(nodus: core.Point, suns: core.Point) -> np.ndarray:
    """Sun altitude and dial bounds select the samples of `suns` whose nodus shadows land on the dial face (`[...]`)."""
    tips = core.shadow(nodus, suns)                                            # [...] Point-
    east, north = core.mv.x & tips, core.mv.y & tips                           # [...] Scalar+, Scalar+
    return ((core.dial & suns) > 0) & (east > east_range[0]) & (east < east_range[1]) \
        & (north > north_range[0]) & (north < north_range[1])


def sunlit_chains(suns: core.Point) -> Topology:
    """Edge chains along the last axis of `suns` restrict to segments where the sun stands above the horizon at both ends."""
    n_samples = suns.shape[-1]
    chains = Topology.chains([n_samples] * (np.prod(suns.shape) // n_samples))
    # Pairing with the dial plane is positive where the sun stands above the horizon at both ends of an edge.
    readings = core.dial & (chains.incidence[1, 0] * suns.reshape(-1).field())   # [2] Scalar+[E]
    return chains.subset((readings > 0).all(axis=0), 1)


def shadow_strokes(nodus: core.Point, suns: core.Point, colours: np.ndarray, width: float, camera: Camera,
                   opacity: float = 1.0) -> np.ndarray:
    """Sunlit shadow chains of `nodus` under `suns` (`[..., n_samples] Point-`) rasterize to an RGBA stroke layer on `camera`."""
    return simplices.strokes(core.shadow(nodus, suns).reshape(-1), sunlit_chains(suns), colours, width, camera,
                             opacity=opacity)


def season_colours(seasons: np.ndarray) -> np.ndarray:
    """Distances from the equinox map `seasons` (`[n_seasons]`) to RGBA colours (`[n_seasons, 4]`) shared across conjugate declinations."""
    return plt.get_cmap(season_map)(0.15 + 0.7 * np.abs(seasons) / np.abs(seasons).max())


def dial_face_layers(ax: plt.Axes, nodus: core.Point, foot: core.Point, seasons: np.ndarray, days: core.Point,
                     suns: core.Point, hours: np.ndarray, faint: bool) -> tuple[Camera, list[np.ndarray]]:
    """Frames `ax` with the seasonal shadow paths, numbered hour lines, and gnomon base, returning `(camera, pictures)`."""
    ax.set(xlim=east_range, ylim=north_range, aspect="equal")
    ax.set_axis_off()
    camera = offset_panel(ax, core.mv.x, core.mv.y)
    # Colour each edge by the season index of its chain.
    season_edges = sunlit_chains(days).elements[1][:, 0] // days.shape[-1]     # [E]
    paths = shadow_strokes(nodus, days, season_colours(seasons)[season_edges, :3], season_width, camera,
                           faint_opacity if faint else 1.0)
    # Draw thin hour lines beneath the seasonal shadow paths.
    lines = shadow_strokes(nodus, suns, np.array(to_rgb(hour_colour)), hour_width, camera)
    hour_lines = core.shadow(nodus, suns)                                      # [..., ...] Point-
    for hour, line, drawn in zip(hours, hour_lines, face_mask(nodus, suns)):
        # Place the numeral just inside the outer end of the visible segment on the dial.
        outer, inner = line[np.flatnonzero(drawn)[[0, -1]]]                    # [] Point-, Point-
        place = outer - inset * (outer - inner).normalized()                   # [] Point-
        ax.text(*(face_planes & place[None]).to_array(), f"{hour}", color=hour_colour, ha="center", va="center",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
    # Draw the gnomon's base from the foot of the style to the point on the dial directly below the nodus.
    below_nodus = nodus - core.up * (core.dial & nodus)                        # [] Point-
    base = simplices.strokes(stack([foot, below_nodus]), Topology.chains([2]), np.array(to_rgb(gnomon_colour)),
                             base_width, camera, cap=simplices.Cap.BUTT)
    return camera, [lines, paths, base]


@charts.style()
def dial_figure(nodus: core.Point, foot: core.Point, seasons: np.ndarray, days: core.Point, suns: core.Point,
                hours: np.ndarray) -> plt.Figure:
    """Draws the horizontal dial face with numbered hour lines and seasonal shadow conics."""
    figure, ax = charts.subplots(460, 345, layout="constrained")
    camera, pictures = dial_face_layers(ax, nodus, foot, seasons, days, suns, hours, faint=False)
    del camera
    overlay(ax, pictures)
    return figure


def daylight_segments(nodus: core.Point, rays: core.Point) -> tuple[core.Point, core.Point]:
    """Daytime `rays` determine sunward ray starts and dial shadow endpoints (`[n_daytime] Point-`, `[n_daytime] Point-`)."""
    daytime = (core.dial & rays) > 0                                           # [...]
    return nodus + sunward * rays[daytime], core.shadow(nodus, rays[daytime])  # [...] Point-, [...] Point-


def sky_view(ax: plt.Axes, nodus: core.Point, envelope_rays: core.Point) -> solids.View:
    """Frames a perspective `View` on `ax` enclosing the dial plate and all daytime ray starts in `envelope_rays`."""
    sunward_points = concatenate([daylight_segments(nodus, day)[0] for day in envelope_rays])  # [...] Point-
    return fitted(solids.view([dial_corners, solids.origin + solids.context.include_from_algebra(sunward_points)], 0.0, elevation, azimuth,
                              margin, 1), ax)


def day_path_ink(nodus: core.Point, foot: core.Point, day: core.Point) -> solids.Ink:
    """Carrying the daily ray cone from the origin to `nodus` defines the inked conic where the dial plate cuts the shadow nappe."""
    # The style runs from its foot to the nodus towards the celestial pole.
    pole = (nodus - foot).normalized()                                         # [] Point-
    to_nodus = solids.translator(solids.context.include_from_algebra(nodus))   # [] Motor+
    # Restricting to degenerate blades reads each point of the renderer's space by its offset from the origin.
    about_origin = solids.context.include_from_algebra(core.cone(pole, day[0]))(solids.Point.restrict.degenerate())  # [] Plane+ <- Point-
    # Keep only the shadow nappe away from the sun across the plane perpendicular to the pole through the nodus.
    across = pole.dual_inverse()                                               # [] Plane+
    shadow_side = solids.half_space(to_nodus >> solids.context.include_from_algebra(across * (across & day[0])))
    return solids.Ink(0, solids.placed(about_origin, to_nodus)[None], shadow_side[None], np.array(to_rgb(day_colour)),
                      path_width, 1.0)


def dial_solids(nodus: core.Point, foot: core.Point) -> list[tuple[solids.Quadric, np.ndarray]]:
    """Quadric slabs represent the horizontal dial plate and the triangular gnomon in 3D space."""
    middle = dial_corners.mean(axis=0) - solids.mv.z.dual() * (plate_thickness / 2)  # [] Point-
    half_extents = np.array([np.ptp(east_range) / 2, np.ptp(north_range) / 2, plate_thickness / 2])
    slab = solids.boxes(solids.translator(middle - solids.origin), half_extents)[None]  # [1, 6] Plane+ <- Point-
    # The triangular gnomon joins the foot of the style, the nodus, and the point on the dial below the nodus.
    below_nodus = nodus - core.up * (core.dial & nodus)                        # [] Point-
    corners = solids.origin + solids.context.include_from_algebra(stack([foot, nodus, below_nodus]))  # [3] Point-
    gnomon = solids.plates(corners[None], gnomon_thickness)                    # [1, 5] Plane+ <- Point-
    return [(slab, np.array(to_rgb(dial_colour))), (gnomon, np.array(to_rgb(gnomon_colour)))]


def sunlight_scene(nodus: core.Point, foot: core.Point, rays: core.Point, day: core.Point,
                   view: solids.View) -> tuple[list[tuple[solids.Quadric, np.ndarray]], list[solids.Ink]]:
    """Daytime sun rods clipped to the vertical walls above the dial plate pair with the inked shadow conic."""
    ray_starts, shadow_points = daylight_segments(nodus, rays)                 # [...] Point-, [...] Point-
    n_daytime = ray_starts.shape[0]
    # Clip rays at the vertical walls standing on the dial edges so rays leaving the plate shorten at its border.
    sides = stack([-solids.mv.x, solids.mv.x, -solids.mv.y, solids.mv.y])      # [4] Plane+
    corners = dial_corners[[0, 2, 0, 2]]                                       # [4] Point-
    walls = solids.half_space(sides - solids.mv.w * (sides & corners))
    rods = solids.rods(solids.origin + solids.context.include_from_algebra(ray_starts),
                       solids.origin + solids.context.include_from_algebra(shadow_points),
                       ray_width / 2 * view.pixel)                             # [n_daytime, 3] Plane+ <- Point-
    beams = concatenate([rods, walls[None].broadcast_to((n_daytime, 4))], axis=-1)  # [n_daytime, 7] Plane+ <- Point-
    beam_colour = ray_opacity * np.array(to_rgb(ray_colour)) + (1 - ray_opacity)
    return [(beams, beam_colour)], [day_path_ink(nodus, foot, day)]


def year_animation(
    nodus: core.Point,
    foot: core.Point,
    seasons: np.ndarray,
    season_days: core.Point,
    suns: core.Point,
    hours: np.ndarray,
    envelope_rays: core.Point,
    states: Iterable[tuple[core.Point, core.Point]],
) -> Iterator[np.ndarray]:
    """Yields RGB frames showing the day's rays past the nodus in space alongside its shadow conic on the dial face."""
    dial = dial_solids(nodus, foot)
    with charts.style():
        figure = plt.figure(figsize=(800 / charts.DPI, 370 / charts.DPI), dpi=charts.DPI)
        sky_axes = figure.add_axes((0.0, 0.0, 0.5, 1.0))
        face = figure.add_axes((0.52, 0.04, 0.46, 0.92))
        camera, pictures = dial_face_layers(face, nodus, foot, seasons, season_days, suns, hours, faint=True)
    path = overlay(face, pictures)
    view = sky_view(sky_axes, nodus, envelope_rays)
    sky = show(sky_axes, np.ones((*view.shape, 3)), view)
    for day_rays, day_suns in states:
        beams, inks = sunlight_scene(nodus, foot, day_rays, day_suns, view)
        sky.set_data(solids.draw(dial + beams, view, np.ones(3), inks, translucent=[], meshes=[]))
        day = shadow_strokes(nodus, day_suns, np.array(to_rgb(day_colour)), day_width, camera)
        path.set_data(drawing.composited(pictures + [day]))
        yield capture(figure)
    plt.close(figure)
