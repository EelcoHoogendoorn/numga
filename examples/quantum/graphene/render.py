"""Drawing for graphene: the pseudospin field, the bands, the pseudospin around the valleys, and the
Berry phase. The bands are drawn on the GPU, the rest by matplotlib."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.quantum.graphene import core
from examples.wgsl import compose, meshes, show, solids

sm = solids.mv
COLOURS = ("#2e86c1", "#7d3c98", "#c0392b", "#d68910")
# One arrow for every few samples of the field.
ARROW_STRIDE = 6
# The bands' chart: the height of one electronvolt against one unit of momentum, the gap between the bands and the
# axes, how far past the bands the axes reach, and the momenta and energies ticked; the axes' colour.
ENERGY_SCALE, AXIS_GAP, AXIS_OVERHANG = 0.3, 0.4, 0.8
MOMENTUM_TICKS, ENERGY_TICKS = np.arange(-4.0, 5.0, 2.0), np.arange(-8.0, 9.0, 4.0)
AXIS_COLOUR = np.full(3, 0.45)
# The view's elevation and azimuth in radians, azimuth zero looking along +y; the margin around the chart as a share
# of the larger side, and the pixels across it.
ELEVATION, AZIMUTH = np.radians(22.0), np.radians(-60.0 + 90.0)
MARGIN, N_PIXELS = 0.1, 1000
# The widths of the axes' shafts, their heads' length and width, and the ticks' radius, in pixels.
AXIS_WIDTH, HEAD_LENGTH, HEAD_WIDTH, TICK_RADIUS = 2.4, 18.0, 10.0, 3.5


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


def draw_field(momenta: core.Vector, field: core.Vector) -> plt.Figure:
    """The pseudospin field over the momentum plane: its length as shade, vanishing at the six
    corners of the zone, and its direction in the plane as arrows, which turn once around each
    corner."""
    k, d = components(momenta), components(field)                                  # [n, n, 3], [n, n, 3]
    length = np.linalg.norm(d, axis=-1)
    sparse = (slice(None, None, ARROW_STRIDE),) * 2
    figure, ax = plt.subplots(figsize=(7, 6))
    shade = ax.pcolormesh(k[..., 0], k[..., 1], length, cmap="viridis", shading="auto")
    ax.quiver(k[sparse][..., 0], k[sparse][..., 1], (d[..., 0] / length)[sparse], (d[..., 1] / length)[sparse],
              color="white", pivot="mid", scale=30)
    ax.set_aspect("equal")
    ax.set_xlabel("momentum x (1/a)")
    ax.set_ylabel("momentum y (1/a)")
    figure.colorbar(shade, ax=ax, label="length of the field (eV)")
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
    # The axes run from the corner just past the bands' front left, along x and up, and along y just past the tip
    # of the x axis.
    reach = np.abs(k[..., :2]).max() + AXIS_GAP
    floor, ceiling = ENERGY_SCALE * (energies.min() - 1), ENERGY_SCALE * (energies.max() + 1)
    corner = np.array([-reach, -reach, floor])
    right = np.array([reach + AXIS_OVERHANG + AXIS_GAP, -reach, floor])
    starts = np.stack([corner, corner, right])
    ends = np.stack([[reach + AXIS_OVERHANG, -reach, floor], [-reach, -reach, ceiling + AXIS_OVERHANG * ENERGY_SCALE],
                     right + [0, 2 * reach + AXIS_OVERHANG, 0]])
    ticks = [corner + np.outer(MOMENTUM_TICKS + reach, [1, 0, 0]), corner + np.outer(ENERGY_SCALE * ENERGY_TICKS - floor, [0, 0, 1]),
             right + np.outer(MOMENTUM_TICKS + reach, [0, 1, 0])]
    view = solids.view([places for places, _ in sheets] + [sm.point(starts), sm.point(ends)], 0.0, ELEVATION, AZIMUTH,
                       MARGIN, N_PIXELS)
    shafts, heads = solids.arrows(sm.point(starts), sm.point(ends), AXIS_WIDTH / 2 * view.pixel,
                                  HEAD_LENGTH * view.pixel, HEAD_WIDTH / 2 * view.pixel)
    tick_balls = solids.ball(sm.point(np.concatenate(ticks)), TICK_RADIUS * view.pixel)[:, None]
    axes = solids.layer([(shafts, AXIS_COLOUR), (heads, AXIS_COLOUR), (tick_balls, AXIS_COLOUR)], view)
    bands = [meshes.layer(places, faces, shade, meshes.smooth(places, faces), view, gloss=0.2)
             for places, shade in sheets]
    figure = plt.figure(figsize=(8, 8 * view.half[1] / view.half[0]))
    ax = figure.add_axes((0, 0, 1, 1))
    show(ax, compose([(*axes, 1.0)] + [(*band, 1.0) for band in bands], np.ones(3)), view)
    # The ticks numbered outside the chart, and the axes named beside their middles.
    for places, numbers, offset, align in zip(ticks, (MOMENTUM_TICKS, ENERGY_TICKS, MOMENTUM_TICKS),
                                              ((-4, -8), (-8, 0), (6, -6)), (("right", "top"), ("right", "center"),
                                                                             ("left", "top"))):
        for place, number in zip(view.across(sm.point(places)), numbers):
            ax.annotate(f"{number:g}", place, xytext=offset, textcoords="offset points", ha=align[0], va=align[1],
                        fontsize=9, color=AXIS_COLOUR)
    middles = np.array([[0, -reach, floor], [-reach, -reach, ceiling + 2 * AXIS_OVERHANG * ENERGY_SCALE],
                        [right[0], 0, floor]])
    for place, name, offset, align in zip(view.across(sm.point(middles)),
                                          ("momentum x (1/a)", "energy (eV)", "momentum y (1/a)"),
                                          ((-14, -26), (8, 0), (34, -34)), ("right", "left", "left")):
        ax.annotate(name, place, xytext=offset, textcoords="offset points", ha=align, va="center", fontsize=10,
                    color=AXIS_COLOUR)
    return figure


def draw_textures(offsets: core.Vector, directions: core.Vector) -> plt.Figure:
    """The pseudospin of the upper band about each valley: its in-plane part as arrows, its part
    along z as colour. It winds once around each valley, in opposite senses."""
    k, s = components(offsets), components(directions)                            # [n, n, 3], [valleys, n, n, 3]
    figure, panels = plt.subplots(1, 2, figsize=(10, 4.8))
    for ax, spin in zip(panels, s):
        arrows = ax.quiver(k[..., 0], k[..., 1], spin[..., 0], spin[..., 1], spin[..., 2], cmap="coolwarm",
                           clim=(-1, 1), pivot="mid", scale=18)
        ax.set_aspect("equal")
        ax.set_xlabel("momentum from the valley, x (1/a)")
    panels[0].set_ylabel("y (1/a)")
    figure.colorbar(arrows, ax=panels, label="pseudospin along z")
    return figure


def draw_berry(radii: np.ndarray, gaps: np.ndarray, rotors: core.Rotor, starts: core.Vector) -> plt.Figure:
    """The Berry phase of a loop against its radius, per gap, for both valleys: pi for every loop
    without a gap, rising from zero toward pi with one, with opposite signs in the two valleys."""
    phases = phase(rotors, starts) / np.pi                                         # [gaps, valleys, radii]
    figure, ax = plt.subplots(figsize=(8, 4.5))
    for gap, pair, colour in zip(gaps, phases, COLOURS):
        ax.plot(radii, pair[0], color=colour, label=f"gap {gap:g} eV")
        ax.plot(radii, pair[1], "--", color=colour)
    ax.axhline(0, color="0.85", linewidth=0.6)
    ax.set_xlabel("loop radius (1/a)")
    ax.set_ylabel("Berry phase (π)")
    ax.legend()
    figure.tight_layout()
    return figure
