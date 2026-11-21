"""Drawing for graphene: the pseudospin field, the bands, the pseudospin around the valleys, and the
Berry phase. The bands are drawn on the GPU, the rest by matplotlib."""

from __future__ import annotations

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects

from numga import dim, stack
from examples.quantum.graphene import core
from examples.util import charts
from numga.topology import Topology
from numga.render import Camera, contours, flow, meshes, simplices, solids
from examples.util.panels import fitted, offset_panel, overlay, pinned, room, show

colours_palette = ("#2e86c1", "#7d3c98", "#c0392b", "#d68910")
# The size of the names on the bands' chart, in points.
font_size = 7
# The pseudospin about the valleys: the side of each valley's panel and the gap between them, in pixels; the colour
# of the streaks of its part in the plane; the diverging map of the part along z, from -1 to 1.
texture_side, texture_gap = 280, 16
streak_colour = np.full(3, 0.15)
texture_map = plt.get_cmap("RdBu_r")
# The streaks of the field's direction over the zone.
field_streak_colour = np.ones(3)
# The bands' chart: the height of one electronvolt against one unit of momentum, and how far past the bands the
# frame's arrows reach, in units of momentum; the frame's colour.
energy_scale = core.mv.scalar(0.3, unit=1 / (dim.length * dim.energy))
axis_overhang = core.mv.scalar(0.8, unit=1 / dim.length)
energy_pad = core.mv.scalar(1.0, unit=dim.energy)
axis_colour = np.full(3, 0.45)
# The colour of the faint cross through the origin of the momentum plane, and of its names.
cross_colour = np.full(3, 0.6)
# The colour of the names at the cross's ends over the pseudospin about the valleys, and the width of their white
# halo, in points.
name_colour, halo_width = "0.25", 2.0
# The view's elevation and azimuth in radians, azimuth zero looking along +y; the margin around the chart as a share
# of the larger side, and the pixels across it, until the view is fitted to its axes.
elevation, azimuth = np.radians(22.0), np.radians(-15.0)
margin, n_pixels = 0.1, 400
# The width of the frame's shafts, and the radius of the balls at their tips, in pixels.
axis_width, tip_radius = 1.6, 3.4


def berry_phases(rotors: core.Rotor, start_directions: core.Pseudospin) -> np.ndarray:
    """Reads each holonomy rotor's signed Berry phase angle relative to the starting pseudospin direction."""
    # A rotor turning a positive angle about the start has minus the start's plane in its bivector part.
    sine_parts = ((-rotors.select[2]) & start_directions).to_array()            # [...]
    return np.arctan2(sine_parts, rotors.select[0].to_array())


def haloed_cross(ax, camera: Camera, horizontal_label: str, vertical_label: str) -> None:
    """Draws a faint origin cross with haloed axis labels over a momentum-plane panel."""
    mv = core.mv
    # The cross's two lines from edge to edge of the panel, the momenta along x and along y at its limits.
    limits = mv.scalar(np.array([ax.get_xlim(), ax.get_ylim()])[..., None], unit=1 / dim.length)  # [2, 2] Scalar+
    ends = (stack((mv.x, mv.y))[:, None] * limits).reshape(-1)                 # [4] Wavevector+
    cross = simplices.strokes(ends, Topology.chains([2, 2]), cross_colour, mv.scalar(charts.FRAME_PX, unit=dim.pixel),
                              camera, cap=simplices.Cap.BUTT)
    overlay(ax, [cross]).set_zorder(1.5)
    halo = [patheffects.withStroke(linewidth=halo_width, foreground="white")]
    ax.annotate(horizontal_label, (1, 0), xycoords=("axes fraction", "data"), xytext=(-3, -3),
                textcoords="offset points", ha="right", va="top", color=name_colour, path_effects=halo, zorder=3)
    ax.annotate(vertical_label, (0, 1), xycoords=("data", "axes fraction"), xytext=(-3, -3),
                textcoords="offset points", ha="right", va="top", color=name_colour, path_effects=halo, zorder=3)


@charts.style()
def field_figure(momenta: core.Wavevector, field: core.Field) -> plt.Figure:
    """Draws the pseudospin field over the momentum plane, shading its magnitude and streaking its in-plane direction."""
    topology = Topology.grid(*momenta.shape)
    mv = momenta.context.multivector
    # The momenta as they are, read by the displacements along x and y.
    places = momenta.reshape(-1)                                                # [...] Wavevector+
    strength = (field.norm() / dim.energy).reshape(-1).to_array()               # [...]
    # The field's direction, its parts in the planes yz and zx, pairings with those planes' patches, drawn as a
    # momentum along x and y.
    parts = field.normalized().reshape(-1)[:, None] & stack((mv.yz.dual(), mv.zx.dual()))   # [..., 2] Scalar+
    directions = (stack((mv.x, mv.y)) * parts).sum(axis=-1)                     # [...] Wavevector+
    # The zone and its six corners, and room for the cross's names.
    figure, ax = charts.subplots(340, 340, layout="constrained")
    # The zone's reach along x and y, read by pairing with the displacements along them.
    readings = stack((mv.x.dual(), mv.y.dual()))[:, None] & places              # [2, ...] Scalar+
    limits = stack((readings.min(axis=-1), readings.max(axis=-1)), axis=-1)     # [2, 2] Scalar+
    ax.set(aspect="equal")
    camera = offset_panel(ax, mv.x.dual(), mv.y.dual(), xlim=limits[0], ylim=limits[1])
    colours = plt.get_cmap("viridis")(strength / strength.max())[:, :3]
    overlay(ax, [contours.picture(places, topology, strength, colours, camera)]).set_zorder(1)
    overlay(ax, [flow.layer(places, topology, directions, camera, camera.across & mv.x, field_streak_colour)]).set_zorder(2)
    haloed_cross(ax, camera, "momentum x", "momentum y")
    return figure


def grid_faces(n_rows: int, n_columns: int) -> np.ndarray:
    """Triangulates a rectangular grid of `n_rows` by `n_columns` points into `[F, 3]` vertex index triples."""
    index = np.arange(n_rows * n_columns).reshape(n_rows, n_columns)
    a, b, c, d = index[:-1, :-1], index[1:, :-1], index[1:, 1:], index[:-1, 1:]
    return np.concatenate([np.stack([a, b, c], axis=-1), np.stack([a, c, d], axis=-1)]).reshape(-1, 3)


def bands_figure(momenta: core.Wavevector, values: core.Scalar) -> plt.Figure:
    """Draws the upper and lower band sheets over the momentum plane with six Dirac cones at the Brillouin-zone corners."""
    mv = momenta.context.multivector
    energies = (values / dim.energy).to_array()                                # [..., 4]
    # The chart's directions: momentum across, along x and y, and energy up, along z.
    horizontal_offsets, vertical_unit = momenta.dual(), mv.z.dual()             # [...] Displacement-, [] Displacement-
    faces = grid_faces(*momenta.shape)
    sheets = []
    for band, cmap in ((-1, "Reds"), (0, "Blues_r")):
        offsets = horizontal_offsets + energy_scale * values[..., band] * vertical_unit   # [...] Displacement-
        # The sheet embedded in the drawing's projective space about its origin.
        places = solids.origin + solids.context.include_from_algebra(offsets).reshape(-1)       # [...] Point-
        # Each band shaded by where its energy lies between its own lowest and highest.
        sheet = energies[..., band]                                             # [...]
        shade = plt.colormaps[cmap]((sheet - sheet.min()) / np.ptp(sheet))[..., :3].reshape(-1, 3)
        sheets.append((places, shade[faces]))
    # The frame from a corner just outside the bands' front left and below them, its arrows reaching just past the
    # bands along x and y, and up past the upper band as the camera sees its far side, raised by the depth across the
    # bands at the view's elevation.
    widths = (stack((mv.x, mv.y)) | momenta[..., None]).abs().reshape(-1)
    half_width = widths.max() + axis_overhang
    levels = values.reshape(-1)
    floor = energy_scale * (levels.min() - energy_pad)
    ceiling = energy_scale * (levels.max() + energy_pad)
    corner = -half_width * (mv.x.dual() + mv.y.dual()) + floor * vertical_unit  # [] Displacement-
    reaches = stack((2 * half_width, 2 * half_width, ceiling - floor + 2 * half_width * np.tan(elevation)))
    tips = corner + stack((mv.x.dual(), mv.y.dual(), vertical_unit)) * reaches  # [3] Displacement-
    frame_corner = solids.origin + solids.context.include_from_algebra(corner)  # [] Point-
    frame_tips = solids.origin + solids.context.include_from_algebra(tips)      # [3] Point-
    view = solids.view([places for places, _ in sheets] + [frame_corner, frame_tips], 0.0, elevation, azimuth, margin,
                       n_pixels)
    figure = plt.figure(figsize=(3.0, 3.0 * view.half[1] / view.half[0]), dpi=200)
    ax = figure.add_axes((0, 0, 1, 1))
    view = fitted(view, ax)
    # The frame's arrows: rods from the corner, each with a ball at its tip.
    shafts = solids.rods(frame_corner, frame_tips, axis_width / 2 * view.pixel)
    heads = solids.ball(frame_tips, tip_radius * view.pixel)[:, None]
    bands = [(meshes.samples(places, faces, shade, meshes.smooth(places, Topology(len(places), faces)), view,
                             face_values=np.zeros(4), colour=meshes.plain, gloss=0.2), 1.0)
             for places, shade in sheets]
    show(ax, solids.draw([(shafts, axis_colour), (heads, axis_colour)], view, np.ones(3), inks=[], translucent=[],
                         meshes=bands), view)
    # Each arrow named just past its tip.
    for place, name, offset, align in zip(view.across(frame_tips), ("momentum x", "momentum y", "energy"),
                                          ((4, 0), (0, 4), (0, 4)), ("left", "center", "center")):
        ax.annotate(name, place, xytext=offset, textcoords="offset points", ha=align, va="center", fontsize=font_size,
                    color=axis_colour)
    return figure


@charts.style()
def textures_figure(pseudospins_at: Callable[[core.Wavevector], core.Pseudospin], extent: core.Scalar) -> plt.Figure:
    """Draws the upper-band pseudospin around both valleys, shading the out-of-plane component and streaking the in-plane winding."""
    width = 2 * texture_side + texture_gap
    figure = plt.figure(figsize=(width / charts.DPI, texture_side / charts.DPI))
    panels = [figure.add_axes((left / width, 0, texture_side / width, 1)) for left in (0, texture_side + texture_gap)]
    n_side = min(room(panels[0]))
    # One momentum at the middle of each pixel, rows down from the top.
    grid_axis = -extent + (np.arange(n_side) + 0.5) * 2 * extent / n_side
    pixels = core.mv.x * grid_axis + core.mv.y * grid_axis[::-1, None]          # [n_side, n_side] Wavevector+
    pseudospins = pseudospins_at(pixels)                                        # [2, n_side, n_side] Pseudospin-
    # The pseudospins' parts along z, their pairings with the plane xy.
    heights = (core.mv.xy & pseudospins).to_array()                             # [2, n_side, n_side]
    # The pseudospins' parts along x and y, their pairings with the planes yz and zx, drawn as directions in the
    # momentum plane along x and y.
    parts = stack((core.mv.yz, core.mv.zx)) & pseudospins[..., None]            # [2, n_side, n_side, 2] Scalar+
    directions = (stack((core.mv.x, core.mv.y)) * parts).sum(axis=-1)           # [2, n_side, n_side] Wavevector+
    # The momenta of the pixels as they are, a grid over the momentum plane.
    topology = Topology.grid(n_side, n_side)
    places = pixels.reshape(-1)                                                 # [n_side * n_side] Wavevector+
    extent_val = float((extent * dim.length).to_array())
    bounds = stack((-extent, extent))
    for ax, height, valley_directions in zip(panels, heights, directions):
        colours = texture_map((height + 1) / 2)[..., :3]
        pinned(ax, np.round(colours * 255).astype(np.uint8), (-extent_val, extent_val, -extent_val, extent_val))
        # The momenta read by the displacements along x and y.
        camera = offset_panel(ax, core.mv.x.dual(), core.mv.y.dual(), xlim=bounds, ylim=bounds)
        # The part in the plane as streaks running downstream, as long as that part: specks where the pseudospin
        # stands along z.
        overlay(ax, [flow.layer(places, topology, valley_directions.reshape(-1), camera,
                                camera.across & core.mv.x, streak_colour)]).set_zorder(2)
        haloed_cross(ax, camera, "momentum x", "momentum y")
    return figure


@charts.style()
def berry_figure(radii: core.Scalar, gaps: core.Scalar, rotors: core.Rotor, start_directions: core.Pseudospin) -> plt.Figure:
    """Plots the signed Berry phase against loop radius across sublattice gaps for both valleys."""
    phases = berry_phases(rotors, start_directions) / np.pi                     # [...]
    radii_values = (radii * dim.length).to_array()
    gap_values = (gaps / dim.energy).to_array()
    # About half the column: a few curves each way, and their legend beside them.
    figure, ax = charts.subplots(440, 240)
    for gap, pair, colour in zip(gap_values, phases, colours_palette):
        ax.plot(radii_values, pair[0], color=colour, label=f"gap {gap:g} eV")
        ax.plot(radii_values, pair[1], "--", color=colour)
    ax.axhline(0, color="black", linewidth=charts.FRAME_WIDTH)
    # The phase read off at none, and at a half turn either way.
    ax.set(xticks=[], yticks=[-1, 0, 1], yticklabels=["−π", "0", "π"], xlabel="loop radius", ylabel="Berry phase")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    figure.tight_layout()
    return figure
