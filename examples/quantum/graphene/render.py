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
from examples.util.wgsl import fitted, meshes, pinned, room, show, solids

sm = solids.mv
COLOURS = ("#2e86c1", "#7d3c98", "#c0392b", "#d68910")
# The size of the names on the bands' chart, in points.
FONT_SIZE = 7
# One arrow for every few samples of the field.
ARROW_STRIDE = 10
# The pseudospin about the valleys: the side of each valley's panel and the gap between them, in pixels; the arrows
# each way, the longest arrow's share of their spacing, their shafts' width as a share of the panel's, and their
# colour; the diverging map of the part along z, from -1 to 1.
TEXTURE_SIDE, TEXTURE_GAP = 280, 16
N_TEXTURE_ARROWS = 7
ARROW_REACH, ARROW_WIDTH = 0.85, 0.012
ARROW_COLOUR = "0.15"
TEXTURE_MAP = plt.get_cmap("RdBu_r")
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


def cross(ax, across: str, up: str) -> None:
    """No scales: a faint cross through the origin over the picture, its lines named at their ends, each along
    itself."""
    ax.axhline(0, color=CROSS_COLOUR, linewidth=charts.FRAME_WIDTH, zorder=1.5)
    ax.axvline(0, color=CROSS_COLOUR, linewidth=charts.FRAME_WIDTH, zorder=1.5)
    ax.annotate(across, (1, 0), xycoords=("axes fraction", "data"), xytext=(3, 0), textcoords="offset points",
                ha="left", va="center", color=CROSS_COLOUR)
    ax.annotate(up, (0, 1), xycoords=("data", "axes fraction"), xytext=(-3, 0), textcoords="offset points",
                ha="right", va="bottom", color=CROSS_COLOUR, rotation=90, rotation_mode="anchor")
    ax.set_axis_off()


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
    corners of the zone, and its direction in the plane as arrows, which turn once around each
    corner."""
    k, d = momentum_coordinates(momenta), plane_components(field)                 # [samples, samples, 2], [samples, samples, 3]
    length = np.linalg.norm(d, axis=-1)
    sparse = (slice(None, None, ARROW_STRIDE),) * 2
    # The zone and its six corners, with an arrow every few samples, and room for the cross's names.
    figure, ax = charts.subplots(340, 340, layout="constrained")
    ax.pcolormesh(k[..., 0], k[..., 1], length, cmap="viridis", shading="auto")
    ax.quiver(k[sparse][..., 0], k[sparse][..., 1], (d[..., 0] / length)[sparse], (d[..., 1] / length)[sparse],
              color="white", pivot="mid", scale=20, zorder=2)
    ax.set_aspect("equal")
    cross(ax, "momentum x", "momentum y")
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
    k = momentum_coordinates(momenta)                                          # [samples, samples, 2]
    energies = values.to_array()                                               # [n, n, 4]
    faces = grid_faces(*k.shape[:2])
    sheets = []
    for sheet, cmap in ((energies[..., -1], "Reds"), (energies[..., 0], "Blues_r")):
        places = sm.point(np.stack([k[..., 0], k[..., 1], ENERGY_SCALE * sheet], axis=-1).reshape(-1, 3))
        # Each band shaded by where its energy lies between its own lowest and highest.
        shade = plt.colormaps[cmap]((sheet - sheet.min()) / np.ptp(sheet))[..., :3].reshape(-1, 3)
        sheets.append((places, shade[faces]))
    # The frame from a corner just outside the bands' front left and below them, its arrows reaching just past the
    # bands along x and y, and up past the upper band as the camera sees its far side, raised by the depth across the
    # bands at the view's elevation.
    half_width = np.abs(k).max() + AXIS_OVERHANG
    floor, ceiling = ENERGY_SCALE * (energies.min() - 1), ENERGY_SCALE * (energies.max() + 1)
    corner = np.array([-half_width, -half_width, floor])
    tips = corner + np.diag([2 * half_width, 2 * half_width, ceiling - floor + 2 * half_width * np.tan(ELEVATION)])   # [3, 3]
    view = solids.view([places for places, _ in sheets] + [sm.point(corner), sm.point(tips)], 0.0, ELEVATION, AZIMUTH,
                       MARGIN, N_PIXELS)
    figure = plt.figure(figsize=(3.0, 3.0 * view.half[1] / view.half[0]), dpi=200)
    ax = figure.add_axes((0, 0, 1, 1))
    view = fitted(view, ax)
    shafts, heads = solids.arrows(sm.point(np.broadcast_to(corner, (3, 3))), sm.point(tips), AXIS_WIDTH / 2 * view.pixel,
                                  HEAD_LENGTH * view.pixel, HEAD_WIDTH / 2 * view.pixel)
    bands = [(meshes.samples(places, faces, shade, meshes.smooth(places, faces), view, face_values=np.zeros(4),
                             colour=meshes.plain, gloss=0.2), 1.0)
             for places, shade in sheets]
    show(ax, solids.draw([(shafts, AXIS_COLOUR), (heads, AXIS_COLOUR)], view, np.ones(3), inks=[], translucent=[],
                         meshes=bands), view)
    # Each arrow named just past its tip.
    for place, name, offset, align in zip(view.across(sm.point(tips)), ("momentum x", "momentum y", "energy"),
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
    # One momentum at the middle of each pixel, rows down from the top, and one at the middle of each cell of the
    # arrows' grid.
    across = -extent + (np.arange(side) + 0.5) * 2 * extent / side
    pixels = core.mv.x * across + core.mv.y * across[::-1, None]                      # [rows, columns] Wavevector+
    centres = -extent + (np.arange(N_TEXTURE_ARROWS) + 0.5) * 2 * extent / N_TEXTURE_ARROWS
    arrows = core.mv.x * centres + core.mv.y * centres[:, None]                       # [arrows, arrows] Wavevector+
    along_z = pseudospin_coordinates(pseudospins_at(pixels))[..., 2]                # [valleys, rows, columns]
    k, s = momentum_coordinates(arrows), pseudospin_coordinates(pseudospins_at(arrows))   # [arrows, arrows, 2], [valleys, arrows, arrows, 3]
    for ax, shade, spin in zip(panels, along_z, s):
        colours = TEXTURE_MAP((shade + 1) / 2)[..., :3]
        pinned(ax, np.round(colours * 255).astype(np.uint8), (-extent, extent, -extent, extent))
        ax.quiver(k[..., 0], k[..., 1], spin[..., 0], spin[..., 1], color=ARROW_COLOUR, pivot="mid",
                  scale=N_TEXTURE_ARROWS / ARROW_REACH, width=ARROW_WIDTH, zorder=2)
        ax.set(xlim=(-extent, extent), ylim=(-extent, extent))
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
