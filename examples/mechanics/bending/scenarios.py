"""Scenes for unsymmetric bending: an I-section, a turned I-section, a Z-section and a slotted tube,
each clamped at one end and loaded straight down at the other; the Z-section with its load
turning once around; and each section pushed along the beam off its centroid, once inside its kern
and once outside."""

from __future__ import annotations

import numpy as np

from examples.mechanics.bending import core
from examples.mechanics.bending.mesh import NAMES, section_meshes

mv, Strain = core.mv, core.Strain
across, up = core.direction([1.0, 0.0]), core.direction([0.0, 1.0])   # points at infinity across the section, and up
# The material's modulus, the beam's length, the pieces it is cut into, and the load at its tip.
MODULUS = 1.0
LENGTH = 10.0
PIECES = 40
LOAD = 4e-5
# The frames of the load's turn.
FRAMES = 48
# The pushes along the beam: their size, and the points they go through, across and up, one inside
# every section's kern and one outside; the half width of the view of the kern and its grid.
PUSH = 1e-3
PUSH_POINTS = [[0.03, 0.04], [0.1, 0.3]]
REACH = 0.75
KERN_GRID = 161


# --- math -----------------------------------------------------------------------------
def cantilever(triangles: core.Point, boundary: core.Point, loads: core.Point) -> core.Beam:
    """A cantilever with its tip load through the centroid, batched over load directions."""
    step = LENGTH / PIECES
    stations = (np.arange(PIECES) + 0.5) * step                             # [PIECES]: the middle of each piece
    rigidity = core.stiffness(triangles, MODULUS)                            # [] Resultant <- Strain
    centre = core.centroid(triangles)                                       # [] Point
    tip = (core.slide * (LENGTH / 2)).exp() >> centre                      # [] Point: the tip of the straight beam
    # Lever arms are measured on the straight reference beam.
    strains = rigidity.solve(core.carried(tip & loads, stations))           # [..., PIECES] Strain
    frames = core.bend(strains, step)                                       # [..., PIECES + 1] Motor
    # Principal bending axes pass through the centroid; the metric does not see the axial slide.
    _, axes = (Strain | Strain).eigh(Strain & rigidity)                    # [3] Strain: weakest first, then the slide
    stress = core.stress(strains[..., 0, None, None], MODULUS, triangles)     # [..., triangles, 3] Scalar
    return core.Beam(triangles, boundary, loads, strains, frames, frames >> centre, axes[:2], stress)


def downward() -> list[core.Beam]:
    """Each section loaded straight down."""
    return [cantilever(triangles, boundary, LOAD * -up) for triangles, boundary in section_meshes()]


def turning() -> core.Beam:
    """The Z-section, its load turning once around in the plane of the section."""
    angles = np.linspace(0.0, 2 * np.pi, FRAMES, endpoint=False)             # [FRAMES]
    loads = (mv.xy * (-angles / 2)).exp() >> (LOAD * -up)                     # [FRAMES] Point: at infinity
    triangles, boundary = section_meshes()[NAMES.index("Z-section")]
    return cantilever(triangles, boundary, loads)


def push_section(triangles: core.Point, points: core.Point, view: core.Point) -> core.Pushes:
    """The stresses of axial pushes, and the corner stretches defining the kern."""
    rigidity = core.stiffness(triangles, MODULUS)                            # [] Resultant <- Strain
    strains = -PUSH * core.pulls(rigidity, points)                          # [pushes] Strain
    stress = core.stress(strains[..., None, None], MODULUS, triangles)       # [pushes, triangles, 3] Scalar
    pulled = core.pulls(rigidity, view)                                     # [rows, columns] Strain
    corners = core.stress(pulled[..., None, None], MODULUS, triangles)       # [rows, columns, triangles, 3] Scalar
    return core.Pushes(triangles, points, strains, stress, view, corners)


def pushes() -> list[core.Pushes]:
    """Each section pushed along the beam through each of the push points."""
    points = core.point(PUSH_POINTS)
    reach = np.linspace(-REACH, REACH, KERN_GRID)
    view = core.point(np.stack(np.meshgrid(reach, reach), axis=-1))
    return [push_section(triangles, points, view) for triangles, boundary in section_meshes()]


if __name__ == "__main__":
    from examples.animation import save_animation, save_figure
    from examples.mechanics.bending import render

    save_figure(render.draw_beams(downward(), NAMES), "bending_sections")
    save_figure(render.draw_pushes(pushes(), NAMES), "bending_kern")
    save_animation(render.animate_turning(turning()), "bending_turning", 60)
