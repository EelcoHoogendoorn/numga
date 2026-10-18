"""Scenes for discrete shells: a disc whose rest lengths grow towards its rim buckles into a saddle, and a
strip with the crossed natural curvature of a seed pod's two layers twists into a ribbon."""

from __future__ import annotations

import numpy as np

from examples.mechanics.shells import core

mv = core.mv
# The disc's radius, its lattice spacing, the share by which its rest lengths grow at the rim, from none
# at the centre, its thickness, and the height of the bumps it starts from.
DISC_RADIUS = 1.0
DISC_SPACING = 0.1
GROWTH = 0.3
DISC_THICKNESS = 0.03
ROUGHNESS = 1e-3
# The strip's length and width, its lattice spacing and thickness, and the natural curvature of the pod's
# two layers, one along each of two directions crossing the strip at the given angle.
STRIP_LENGTH = 2.0
STRIP_WIDTH = 0.3
STRIP_SPACING = 0.08
STRIP_THICKNESS = 0.05
NATURAL_CURVATURE = 6.0
POD_ANGLE = np.pi / 4
# The relaxations: the step on either lattice, the share of velocity lost every step, the steps, and every
# how many a frame.
DISC_STEP = 0.02
STRIP_STEP = 0.012
DAMPING = 0.01
STEPS = 2000
EVERY = 25
SEED = 0


# --- math -----------------------------------------------------------------------------
def disc() -> tuple[np.ndarray, list[core.Point]]:
    """The growing disc's faces and its points at every frame `[frames] Point[V]`."""
    flat = core.lattice(DISC_SPACING, DISC_RADIUS)
    flat = flat[np.linalg.norm(flat, axis=-1) <= DISC_RADIUS + 1e-9]
    mesh = core.flat_mesh(flat, DISC_SPACING)
    tails, heads = (end * mesh.vertices for end in mesh.edge_ends)               # Point[E] each
    # Each edge grows with the square of its middle's distance from the centre, in disc radii.
    distance = (mv.w.dual() & ((tails + heads) / 2)).norm() / DISC_RADIUS       # Scalar[E]
    lengths = (tails & heads).norm() * (1 + GROWTH * distance * distance)       # Scalar[E]
    sheet = core.sheet(mesh, lengths, 0 * (core.Line | core.Line), mv.z, DISC_THICKNESS)
    frames = list(core.relax(sheet, roughened(flat), DISC_STEP, DAMPING, STEPS, EVERY))

    # --- checks
    settled = frames[-1]
    # The forces inside the sheet balance: their lines add up to no force and no moment.
    total = core.forque(settled, core.forces(sheet, settled))
    assert (-(total | total) - (total.dual() | total.dual())).to_array() < 1e-30
    # The disc has left its plane and settled close to its grown lengths.
    assert np.ptp((settled & mv.z).batch().to_array()) > 0.3
    tails, heads = (end * settled for end in mesh.edge_ends)
    strain = ((tails & heads).norm() / lengths - 1).batch().to_array()
    assert np.median(np.abs(strain)) < 0.02
    return mesh.faces, frames


def strip() -> tuple[np.ndarray, list[core.Point]]:
    """The pod strip's faces and its points at every frame `[frames] Point[V]`."""
    flat = core.lattice(STRIP_SPACING, STRIP_LENGTH)
    inside = (np.abs(flat[:, 0]) <= STRIP_LENGTH / 2) & (np.abs(flat[:, 1]) <= STRIP_WIDTH / 2)
    mesh = core.flat_mesh(flat[inside], STRIP_SPACING)
    tails, heads = (end * mesh.vertices for end in mesh.edge_ends)               # Point[E] each
    lengths = (tails & heads).norm()                                             # Scalar[E]
    # The two layers curve along lines through the centre, turned either way from the strip's axis.
    axis = mv.w.dual() & mv.x.dual()                                               # [] Line
    along, across = (((mv.xy * (-sense * POD_ANGLE / 2)).exp() >> axis) for sense in (1, -1))
    curvature = NATURAL_CURVATURE * ((across | core.Line) * (across | core.Line) - (along | core.Line) * (along | core.Line))
    sheet = core.sheet(mesh, lengths, curvature, mv.z, STRIP_THICKNESS)
    frames = list(core.relax(sheet, roughened(flat[inside]), STRIP_STEP, DAMPING, STEPS, EVERY))

    # --- checks
    settled = frames[-1]
    total = core.forque(settled, core.forces(sheet, settled))
    assert (-(total | total) - (total.dual() | total.dual())).to_array() < 1e-30
    # The strip has left its plane: neither layer's curvature can win along its whole length, and the
    # strip twists between them.
    assert np.ptp((settled & mv.z).batch().to_array()) > 0.1
    return mesh.faces, frames


# --- plumbing -------------------------------------------------------------------------
def roughened(flat: np.ndarray) -> core.Point:
    """The points of a flat sheet `[V, 2]`, lifted by small random heights to let it leave its plane."""
    heights = ROUGHNESS * np.random.default_rng(SEED).normal(size=(len(flat), 1))
    return mv.point(np.concatenate([flat, heights], axis=-1)).field()


if __name__ == "__main__":
    from examples.animation import save_animation, save_figure
    from examples.mechanics.shells import render

    disc_faces, disc_frames = disc()
    strip_faces, strip_frames = strip()
    save_figure(render.draw_settled([(disc_faces, disc_frames[-1]), (strip_faces, strip_frames[-1])]), "shells_settled")
    save_animation(render.animate(disc_faces, disc_frames), "shells_disc", render.DURATION_MS)
    save_animation(render.animate(strip_faces, strip_frames), "shells_strip", render.DURATION_MS)
