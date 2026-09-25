"""Drawing for graphene: the pseudospin field, the bands, the pseudospin around the valleys, and the
Berry phase."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.quantum.graphene import core

COLOURS = ("#2e86c1", "#7d3c98", "#c0392b", "#d68910")
# One arrow for every few samples of the field.
ARROW_STRIDE = 6


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


def draw_bands(momenta: core.Vector, values: core.Scalar) -> plt.Figure:
    """The upper and lower bands over the momentum plane: six cones at the corners of the zone."""
    k = components(momenta)
    energies = values.to_array()
    figure = plt.figure(figsize=(8, 6.5))
    ax = figure.add_subplot(projection="3d")
    for sheet, cmap in ((energies[..., -1], "Reds"), (energies[..., 0], "Blues_r")):
        ax.plot_surface(k[..., 0], k[..., 1], sheet, cmap=cmap, linewidth=0, antialiased=True, alpha=0.9,
                        rcount=120, ccount=120)
    ax.set_xlabel("momentum x (1/a)")
    ax.set_ylabel("momentum y (1/a)")
    ax.set_zlabel("energy (eV)")
    ax.view_init(elev=22, azim=-60)
    ax.set_title("graphene's bands: cones where they meet")
    figure.tight_layout()
    return figure


def draw_textures(offsets: core.Vector, directions: core.Vector) -> plt.Figure:
    """The pseudospin of the upper band about each valley: its in-plane part as arrows, its part
    along z as colour. It winds once around each valley, in opposite senses."""
    k, s = components(offsets), components(directions)                            # [n, n, 3], [valleys, n, n, 3]
    figure, panels = plt.subplots(1, 2, figsize=(10, 4.8))
    for ax, spin, name in zip(panels, s, ("valley K", "valley K'")):
        arrows = ax.quiver(k[..., 0], k[..., 1], spin[..., 0], spin[..., 1], spin[..., 2], cmap="coolwarm",
                           clim=(-1, 1), pivot="mid", scale=18)
        ax.set_aspect("equal")
        ax.set_title(name)
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
    ax.set_title("solid: valley K, dashed: valley K'")
    ax.legend()
    figure.tight_layout()
    return figure
