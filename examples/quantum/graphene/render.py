"""Drawing for graphene: the pseudospin field, the bands, the pseudospin around the valleys, and the
Berry phase. The bands are drawn on the GPU, the rest by matplotlib."""

from __future__ import annotations

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects

from numga import stack
from examples.quantum.graphene import core
from examples.util import charts
from examples.util.wgsl import contours, fitted, flow, meshes, pinned, room, show, solids

COLOURS = ("#2e86c1", "#7d3c98", "#c0392b", "#d68910")
# The size of the names on the bands' chart, in points.
FONT_SIZE = 7
# The pseudospin about the valleys: the side of each valley's panel and the gap between them, in pixels; the colour
# of the streaks of its part in the plane; the diverging map of the part along z, from -1 to 1.
TEXTURE_SIDE, TEXTURE_GAP = 280, 16
STREAK_COLOUR = np.full(3, 0.15)
TEXTURE_MAP = plt.get_cmap("RdBu_r")
# The streaks of the field's direction over the zone.
FIELD_STREAK_COLOUR = np.ones(3)
# The bands' chart: the height of one electronvolt against one unit of momentum, and how far past the bands the
# frame's arrows reach, in units of momentum; the frame's colour.
ENERGY_SCALE, AXIS_OVERHANG = 0.3, 0.8
AXIS_COLOUR = np.full(3, 0.45)
# The colour of the faint cross through the origin of the momentum plane, and of its names.
CROSS_COLOUR = "0.6"
# The colour of the names at the cross's ends over the pseudospin about the valleys, and the width of their white
# halo, in points.
NAME_COLOUR, HALO_WIDTH = "0.25", 2.0
# The view's elevation and azimuth in radians, azimuth zero looking along +y; the margin around the chart as a share
# of the larger side, and the pixels across it, until the view is fitted to its axes.
ELEVATION, AZIMUTH = np.radians(22.0), np.radians(-15.0)
MARGIN, N_PIXELS = 0.1, 400
# The widths of the frame's shafts, and their heads' length and width, in pixels.
AXIS_WIDTH, HEAD_LENGTH, HEAD_WIDTH = 1.6, 12.0, 6.8


def momentum_coordinates(momenta: core.Wavevector) -> np.ndarray:
    """The x and y coordinates of momenta: their pairings with the displacements along x and y."""
    mv = momenta.context.multivector
    return (stack((mv.x.dual(), mv.y.dual())) & momenta[..., None]).to_array()


def pseudospin_coordinates(pseudospins: core.Pseudospin) -> np.ndarray:
    """The x, y and z parts of pseudospins: their pairings with the planes yz, zx and xy."""
    mv = pseudospins.context.multivector
    return (stack((mv.yz, mv.zx, mv.xy)) & pseudospins[..., None]).to_array()


def plane_components(fields: core.Field) -> np.ndarray:
    """The parts of fields in the planes yz, zx and xy: their pairings with those planes' patches."""
    mv = fields.context.multivector
    return (fields[..., None] & stack((mv.yz.dual(), mv.zx.dual(), mv.xy.dual()))).to_array()


def phase(rotors: core.Rotor, starts: core.Pseudospin) -> np.ndarray:
    """The Berry phase of each holonomy: its rotor's angle, signed by the sense of the turn about the
    pseudospin direction the loop starts from."""
    # A rotor turning a positive angle about the start has minus the start's plane in its bivector part.
    turned = ((-rotors.select[2]) & starts).to_array()
    return np.arctan2(turned, rotors.select[0].to_array())


def haloed_cross(ax, across: str, up: str) -> None:
    """No scales: a faint cross through the origin over the picture, its lines named inside the picture near their
    ends, each name on a thin white halo that keeps it legible over the colours and the arrows."""
    ax.axhline(0, color=CROSS_COLOUR, linewidth=charts.FRAME_WIDTH, zorder=1.5)
    ax.axvline(0, color=CROSS_COLOUR, linewidth=charts.FRAME_WIDTH, zorder=1.5)
    halo = [patheffects.withStroke(linewidth=HALO_WIDTH, foreground="white")]
    ax.annotate(across, (1, 0), xycoords=("axes fraction", "data"), xytext=(-3, -3), textcoords="offset points",
                ha="right", va="top", color=NAME_COLOUR, path_effects=halo, zorder=3)
    ax.annotate(up, (0, 1), xycoords=("data", "axes fraction"), xytext=(-3, -3), textcoords="offset points",
                ha="right", va="top", color=NAME_COLOUR, path_effects=halo, zorder=3)


@charts.style()
def draw_field(momenta: core.Wavevector, field: core.Field) -> plt.Figure:
    """The pseudospin field over the momentum plane: its length as shade, vanishing at the six
    corners of the zone, and its direction in the plane as streaks running downstream, which turn once around each
    corner."""
    k = momentum_coordinates(momenta)                                           # [samples, samples, 2]
    mesh = meshes.Mesh.grid(*momenta.shape)
    # The momentum plane drawn as a plane of space: each momentum at the place of its dual.
    places = solids.origin + solids.context.include_from_algebra(momenta.dual().reshape(-1))   # [samples] Point-
    strength = field.norm().reshape(-1).to_array()                              # [samples]
    # The field's direction, its parts in the planes yz and zx drawn as a direction in the momentum plane.
    d = plane_components(field.normalized()).reshape(-1, 3)                    # [samples, 3]
    directions = solids.mv.direction(d * np.array([1, 1, 0]))                  # [samples] Point-
    # The zone and its six corners, and room for the cross's names.
    figure, ax = charts.subplots(340, 340, layout="constrained")
    ax.set(xlim=(k[..., 0].min(), k[..., 0].max()), ylim=(k[..., 1].min(), k[..., 1].max()), aspect="equal")
    colours = plt.get_cmap("viridis")(strength / strength.max())[:, :3]
    contours.overlay(ax, places, mesh, strength, colours).set_zorder(1)
    flow.overlay(ax, places, mesh, directions, 1.0, FIELD_STREAK_COLOUR).set_zorder(2)
    haloed_cross(ax, "momentum x", "momentum y")
    return figure


def grid_faces(rows: int, columns: int) -> np.ndarray:
    """The triangles of a grid of points, two to a cell, indices into the grid flattened."""
    index = np.arange(rows * columns).reshape(rows, columns)
    a, b, c, d = index[:-1, :-1], index[1:, :-1], index[1:, 1:], index[:-1, 1:]
    return np.concatenate([np.stack([a, b, c], axis=-1), np.stack([a, c, d], axis=-1)]).reshape(-1, 3)


def draw_bands(momenta: core.Wavevector, values: core.Scalar) -> plt.Figure:
    """The upper and lower bands over the momentum plane: six cones at the corners of the zone. Each band is a sheet
    shaded by its own colour map, energy up, with axes of momentum along the front edges and of energy up the corner
    between them."""
    mv = momenta.context.multivector
    energies = values.to_array()                                               # [samples, samples, modes]
    # The chart's directions: momentum across, along x and y, and energy up, along z.
    across, up = momenta.dual(), mv.z.dual()                                    # [samples, samples] Place-, [] Place-
    faces = grid_faces(*momenta.shape)
    sheets = []
    for band, cmap in ((-1, "Reds"), (0, "Blues_r")):
        offsets = across + ENERGY_SCALE * values[..., band] * up                 # [samples, samples] Place-
        places = solids.origin + solids.context.include_from_algebra(offsets).reshape(-1)       # [samples * samples] Point-
        # Each band shaded by where its energy lies between its own lowest and highest.
        sheet = energies[..., band]                                             # [samples, samples]
        shade = plt.colormaps[cmap]((sheet - sheet.min()) / np.ptp(sheet))[..., :3].reshape(-1, 3)
        sheets.append((places, shade[faces]))
    # The frame from a corner just outside the bands' front left and below them, its arrows reaching just past the
    # bands along x and y, and up past the upper band as the camera sees its far side, raised by the depth across the
    # bands at the view's elevation.
    widths = (stack((mv.x, mv.y)) | momenta[..., None]).abs().reshape(-1)
    half_width = widths.max() + AXIS_OVERHANG
    levels = values.reshape(-1)
    floor = ENERGY_SCALE * (levels.min() - 1)
    ceiling = ENERGY_SCALE * (levels.max() + 1)
    corner = -half_width * (mv.x.dual() + mv.y.dual()) + floor * up            # [] Place-
    reaches = stack((2 * half_width, 2 * half_width, ceiling - floor + 2 * half_width * np.tan(ELEVATION)))
    tips = corner + stack((mv.x.dual(), mv.y.dual(), up)) * reaches            # [3] Place-
    frame_corner = solids.origin + solids.context.include_from_algebra(corner)                   # [] Point-
    frame_tips = solids.origin + solids.context.include_from_algebra(tips)                       # [3] Point-
    view = solids.view([places for places, _ in sheets] + [frame_corner, frame_tips], 0.0, ELEVATION, AZIMUTH, MARGIN,
                       N_PIXELS)
    figure = plt.figure(figsize=(3.0, 3.0 * view.half[1] / view.half[0]), dpi=200)
    ax = figure.add_axes((0, 0, 1, 1))
    view = fitted(view, ax)
    shafts, heads = solids.arrows(frame_corner, frame_tips, AXIS_WIDTH / 2 * view.pixel, HEAD_LENGTH * view.pixel,
                                  HEAD_WIDTH / 2 * view.pixel)
    bands = [(meshes.samples(places, faces, shade, meshes.smooth(places, meshes.Mesh(len(places), faces)), view, face_values=np.zeros(4),
                             colour=meshes.plain, gloss=0.2), 1.0)
             for places, shade in sheets]
    show(ax, solids.draw([(shafts, AXIS_COLOUR), (heads, AXIS_COLOUR)], view, np.ones(3), inks=[], translucent=[],
                         meshes=bands), view)
    # Each arrow named just past its tip.
    for place, name, offset, align in zip(view.across(frame_tips), ("momentum x", "momentum y", "energy"),
                                          ((4, 0), (0, 4), (0, 4)), ("left", "center", "center")):
        ax.annotate(name, place, xytext=offset, textcoords="offset points", ha=align, va="center", fontsize=FONT_SIZE,
                    color=AXIS_COLOUR)
    return figure


@charts.style()
def draw_textures(pseudospins_at: Callable[[core.Wavevector], core.Pseudospin], extent: float) -> plt.Figure:
    """The pseudospin of the upper band about each valley, over a square of momenta of half width `extent`: its part
    along z as the colour of every pixel, and its part in the plane as arrows on a coarser grid. It winds once around
    each valley, in opposite senses."""
    width = 2 * TEXTURE_SIDE + TEXTURE_GAP
    figure = plt.figure(figsize=(width / charts.DPI, TEXTURE_SIDE / charts.DPI))
    panels = [figure.add_axes((left / width, 0, TEXTURE_SIDE / width, 1)) for left in (0, TEXTURE_SIDE + TEXTURE_GAP)]
    side = min(room(panels[0]))
    # One momentum at the middle of each pixel, rows down from the top.
    across = -extent + (np.arange(side) + 0.5) * 2 * extent / side
    pixels = core.mv.x * across + core.mv.y * across[::-1, None]                      # [rows, columns] Wavevector+
    spins = pseudospin_coordinates(pseudospins_at(pixels))                          # [valleys, rows, columns, 3]
    # The momentum plane drawn as a plane of space, and the pixels as a grid over it.
    mesh = meshes.Mesh.grid(side, side)
    places = solids.origin + solids.context.include_from_algebra(pixels.dual().reshape(-1))   # [pixels] Point-
    for ax, spin in zip(panels, spins):
        colours = TEXTURE_MAP((spin[..., 2] + 1) / 2)[..., :3]
        pinned(ax, np.round(colours * 255).astype(np.uint8), (-extent, extent, -extent, extent))
        ax.set(xlim=(-extent, extent), ylim=(-extent, extent))
        # The part in the plane as streaks running downstream, as long as that part: specks where the pseudospin
        # stands along z.
        directions = solids.mv.direction(spin.reshape(-1, 3) * np.array([1, 1, 0]))   # [pixels] Point-
        flow.overlay(ax, places, mesh, directions, 1.0, STREAK_COLOUR).set_zorder(2)
        haloed_cross(ax, "momentum x", "momentum y")
    return figure


@charts.style()
def draw_berry(radii: np.ndarray, gaps: np.ndarray, rotors: core.Rotor, starts: core.Pseudospin) -> plt.Figure:
    """The Berry phase of a loop against its radius, per gap, for both valleys: pi for every loop
    without a gap, rising from zero toward pi with one, with opposite signs in the two valleys."""
    phases = phase(rotors, starts) / np.pi                                         # [gaps, valleys, radii]
    # About half the column: a few curves each way, and their legend beside them.
    figure, ax = charts.subplots(440, 240)
    for gap, pair, colour in zip(gaps, phases, COLOURS):
        ax.plot(radii, pair[0], color=colour, label=f"gap {gap:g} eV")
        ax.plot(radii, pair[1], "--", color=colour)
    ax.axhline(0, color="black", linewidth=charts.FRAME_WIDTH)
    # The phase read off at none, and at a half turn either way.
    ax.set(xticks=[], yticks=[-1, 0, 1], yticklabels=["−π", "0", "π"], xlabel="loop radius", ylabel="Berry phase")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    figure.tight_layout()
    return figure
