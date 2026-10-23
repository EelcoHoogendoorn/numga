"""Drawing for graphene: the pseudospin field, the bands, the pseudospin around the valleys, and the
Berry phase. The bands are drawn on the GPU, the rest by matplotlib."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.quantum.graphene import core
from examples.util.wgsl import fitted, meshes, show, solids

sm = solids.mv
COLOURS = ("#2e86c1", "#7d3c98", "#c0392b", "#d68910")
# One arrow for every few samples of the field.
ARROW_STRIDE = 6
# The bands' chart: the height of one electronvolt against one unit of momentum, and how far past the bands the
# frame's arrows reach, in units of momentum; the frame's colour.
ENERGY_SCALE, AXIS_OVERHANG = 0.3, 0.8
AXIS_COLOUR = np.full(3, 0.45)
# The colour of the faint cross through the origin of the momentum plane, and of its names.
CROSS_COLOUR = "0.6"
# The view's elevation and azimuth in radians, azimuth zero looking along +y; the margin around the chart as a share
# of the larger side, and the pixels across it, until the view is fitted to its axes.
ELEVATION, AZIMUTH = np.radians(22.0), np.radians(-15.0)
MARGIN, N_PIXELS = 0.1, 1000
# The widths of the frame's shafts, and their heads' length and width, in pixels.
AXIS_WIDTH, HEAD_LENGTH, HEAD_WIDTH = 1.92, 14.4, 8.0


def components(vectors: core.Vector) -> np.ndarray:
    """The x, y and z components of vectors."""
    return vectors.cast(core.ga.subspace("x y z")).kernel


def phase(rotors: core.Rotor, starts: core.Vector) -> np.ndarray:
    """The Berry phase of each holonomy: its rotor's angle, signed by the sense of the turn about the
    pseudospin direction the loop starts from."""
    # The plane perpendicular to the start.
    about = core.mv.xyz * starts                                                   # [...] Bivector
    turned = (rotors.select_subspace(core.ga.subspace.bivector()) | about).to_array()
    return np.arctan2(turned, rotors.select[0].to_array())


def cross(ax, across: str, up: str) -> None:
    """No scales: a faint cross through the origin over the picture, its lines named at their ends, each along
    itself."""
    ax.axhline(0, color=CROSS_COLOUR, linewidth=0.7, zorder=1.5)
    ax.axvline(0, color=CROSS_COLOUR, linewidth=0.7, zorder=1.5)
    ax.annotate(across, (1, 0), xycoords=("axes fraction", "data"), xytext=(4, 0), textcoords="offset points",
                ha="left", va="center", fontsize=10, color=CROSS_COLOUR)
    ax.annotate(up, (0, 1), xycoords=("data", "axes fraction"), xytext=(-4, 0), textcoords="offset points",
                ha="right", va="bottom", fontsize=10, color=CROSS_COLOUR, rotation=90, rotation_mode="anchor")
    ax.set_axis_off()


def draw_field(momenta: core.Vector, field: core.Vector) -> plt.Figure:
    """The pseudospin field over the momentum plane: its length as shade, vanishing at the six
    corners of the zone, and its direction in the plane as arrows, which turn once around each
    corner."""
    k, d = components(momenta), components(field)                                  # [n, n, 3], [n, n, 3]
    length = np.linalg.norm(d, axis=-1)
    sparse = (slice(None, None, ARROW_STRIDE),) * 2
    figure, ax = plt.subplots(figsize=(6, 6), layout="constrained")
    ax.pcolormesh(k[..., 0], k[..., 1], length, cmap="viridis", shading="auto")
    ax.quiver(k[sparse][..., 0], k[sparse][..., 1], (d[..., 0] / length)[sparse], (d[..., 1] / length)[sparse],
              color="white", pivot="mid", scale=30, zorder=2)
    ax.set_aspect("equal")
    cross(ax, "momentum x", "momentum y")
    return figure


def grid_faces(rows: int, columns: int) -> np.ndarray:
    """The triangles of a grid of points, two to a cell, indices into the grid flattened."""
    index = np.arange(rows * columns).reshape(rows, columns)
    a, b, c, d = index[:-1, :-1], index[1:, :-1], index[1:, 1:], index[:-1, 1:]
    return np.concatenate([np.stack([a, b, c], axis=-1), np.stack([a, c, d], axis=-1)]).reshape(-1, 3)


def draw_bands(momenta: core.Vector, values: core.Scalar) -> plt.Figure:
    """The upper and lower bands over the momentum plane: six cones at the corners of the zone. Each band is a sheet
    shaded by its own colour map, energy up, with axes of momentum along the front edges and of energy up the corner
    between them."""
    k = components(momenta)                                                    # [n, n, 3]
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
    reach = np.abs(k[..., :2]).max() + AXIS_OVERHANG
    floor, ceiling = ENERGY_SCALE * (energies.min() - 1), ENERGY_SCALE * (energies.max() + 1)
    corner = np.array([-reach, -reach, floor])
    tips = corner + np.diag([2 * reach, 2 * reach, ceiling - floor + 2 * reach * np.tan(ELEVATION)])   # [3, 3]
    view = solids.view([places for places, _ in sheets] + [sm.point(corner), sm.point(tips)], 0.0, ELEVATION, AZIMUTH,
                       MARGIN, N_PIXELS)
    figure = plt.figure(figsize=(8, 8 * view.half[1] / view.half[0]))
    ax = figure.add_axes((0, 0, 1, 1))
    view = fitted(view, ax)
    shafts, heads = solids.arrows(sm.point(np.broadcast_to(corner, (3, 3))), sm.point(tips), AXIS_WIDTH / 2 * view.pixel,
                                  HEAD_LENGTH * view.pixel, HEAD_WIDTH / 2 * view.pixel)
    bands = [(meshes.samples(places, faces, shade, meshes.smooth(places, faces), view, gloss=0.2), 1.0)
             for places, shade in sheets]
    show(ax, solids.draw([(shafts, AXIS_COLOUR), (heads, AXIS_COLOUR)], view, np.ones(3), meshes=bands), view)
    # Each arrow named just past its tip.
    for place, name, offset, align in zip(view.across(sm.point(tips)), ("momentum x", "momentum y", "energy"),
                                          ((8, 0), (0, 8), (0, 8)), ("left", "center", "center")):
        ax.annotate(name, place, xytext=offset, textcoords="offset points", ha=align, va="center", fontsize=10,
                    color=AXIS_COLOUR)
    return figure


def draw_textures(offsets: core.Vector, directions: core.Vector) -> plt.Figure:
    """The pseudospin of the upper band about each valley: its in-plane part as arrows, its part
    along z as colour. It winds once around each valley, in opposite senses."""
    k, s = components(offsets), components(directions)                            # [n, n, 3], [valleys, n, n, 3]
    figure, panels = plt.subplots(1, 2, figsize=(10, 4.8), layout="constrained")
    for ax, spin in zip(panels, s):
        ax.quiver(k[..., 0], k[..., 1], spin[..., 0], spin[..., 1], spin[..., 2], cmap="coolwarm", clim=(-1, 1),
                  pivot="mid", scale=18, zorder=2)
        ax.set_aspect("equal")
        cross(ax, "momentum x", "momentum y")
    return figure


def draw_berry(radii: np.ndarray, gaps: np.ndarray, rotors: core.Rotor, starts: core.Vector) -> plt.Figure:
    """The Berry phase of a loop against its radius, per gap, for both valleys: pi for every loop
    without a gap, rising from zero toward pi with one, with opposite signs in the two valleys."""
    phases = phase(rotors, starts) / np.pi                                         # [gaps, valleys, radii]
    figure, ax = plt.subplots(figsize=(8, 4.5))
    for gap, pair, colour in zip(gaps, phases, COLOURS):
        ax.plot(radii, pair[0], color=colour, label=f"gap {gap:g} eV")
        ax.plot(radii, pair[1], "--", color=colour)
    ax.axhline(0, color="black", linewidth=0.8)
    # The phase read off at none, and at a half turn either way.
    ax.set(xticks=[], yticks=[-1, 0, 1], yticklabels=["−π", "0", "π"], xlabel="loop radius", ylabel="Berry phase")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    figure.tight_layout()
    return figure
