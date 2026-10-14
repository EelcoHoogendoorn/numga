"""Scenes for Euler–Bernoulli beam bending: an I-section, a turned I-section, a Z-section and a slotted
tube, each clamped at one end and loaded straight down at the other; the Z-section with its load turning
once around; and each section pushed along the beam off its centroid, once inside its kern and once
outside."""

from __future__ import annotations

import numpy as np

from examples.mechanics.bending import core
from examples.mechanics.bending.mesh import NAMES, section_meshes

mv, Strain = core.mv, core.Strain
# The point at infinity up the section.
up_direction = mv.y.dual()                                                   # [] Point
# The material's modulus, the beam's length, the pieces it is cut into, and the load at its tip.
MODULUS = 1.0
LENGTH = 10.0
PIECES = 40
LOAD = 4e-5
# The frames of the load's turn, and the duration of each.
FRAMES = 48
DURATION_MS = 60
# The pushes along the beam: their size, and the points they go through, across and up, one inside
# every section's kern and one outside; the half width of the view of the kern and its grid.
PUSH = 1e-3
PUSH_POINTS = [[0.03, 0.04], [0.1, 0.3]]
REACH = 0.75
KERN_GRID = 161


# --- math -----------------------------------------------------------------------------
def cantilever(triangles: core.Point, loads: core.Point) -> tuple[core.Strain, core.Motor, core.Strain, core.Scalar]:
    """A cantilever with its tip load through the centroid, batched over loads: the strain of each piece
    `[..., pieces]`, the frame of each cross-section `[..., pieces + 1]`, the principal bending axes `[2]`,
    and the stress at the clamp at the corners of the section's triangles `[..., triangles, 3]`."""
    step = LENGTH / PIECES
    # The middle of each piece.
    stations = (np.arange(PIECES) + 0.5) * step                             # [pieces]
    rigidity = core.stiffness(triangles, MODULUS)                            # [] Resultant <- Strain
    # The centroid of the whole section, and the tip of the unbent beam: the centroid shifted along the
    # beam by the beam's length.
    centre = core.centroid(triangles)                                       # [] Point
    tip = (core.shift_along_beam * (LENGTH / 2)).exp() >> centre            # [] Point
    # Lever arms are measured on the straight reference beam.
    strains = rigidity.solve(core.carried(tip & loads, stations))           # [..., pieces] Strain
    frames = core.bend(strains, step)                                       # [..., pieces + 1] Motor
    # Principal bending axes pass through the centroid; the metric does not see the shift along the
    # beam. Weakest first, then the shift.
    _, axes = (Strain | Strain).eigh(Strain & rigidity)                    # [3] Strain
    stress = core.stress(strains[..., 0, None, None], MODULUS, triangles)  # [..., triangles, 3] Scalar
    return strains, frames, axes[:2], stress


def turning_loads() -> core.Point:
    """The tip load turning once around the beam, in the plane of the section."""
    angles = np.linspace(0, 2 * np.pi, FRAMES, endpoint=False)               # [frames]
    return (mv.xy * (-angles / 2)).exp() >> (LOAD * -up_direction)           # [frames] Point


def kern_view() -> core.Point:
    """The points of the section over which the kern is drawn."""
    reach = np.linspace(-REACH, REACH, KERN_GRID)
    return core.point(np.stack(np.meshgrid(reach, reach), axis=-1))          # [rows, columns] Point


def push_section(triangles: core.Point, points: core.Point, view: core.Point
                 ) -> tuple[core.Strain, core.Scalar, core.Scalar]:
    """Pushes along the beam through points of a section: their strains `[pushes]` and stresses at the
    corners of the section's triangles `[pushes, triangles, 3]`; and, for a unit pull through each point of
    the view, the stretch at those corners `[rows, columns, triangles, 3]`."""
    rigidity = core.stiffness(triangles, MODULUS)                            # [] Resultant <- Strain
    strains = -PUSH * core.pulls(rigidity, points)                          # [pushes] Strain
    stress = core.stress(strains[..., None, None], MODULUS, triangles)       # [pushes, triangles, 3] Scalar
    pulled = core.pulls(rigidity, view)                                     # [rows, columns] Strain
    corners = core.stress(pulled[..., None, None], MODULUS, triangles)       # [rows, columns, triangles, 3] Scalar
    return strains, stress, corners


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation, save_figure
    from examples.mechanics.bending import render

    sections = section_meshes()
    beams = [cantilever(triangles, LOAD * -up_direction) for triangles, _ in sections]
    triangles, boundary = sections[NAMES.index("Z-section")]
    loads = turning_loads()
    turning = cantilever(triangles, loads)
    points, view = core.point(PUSH_POINTS), kern_view()                       # [pushes], [rows, columns] Point
    pushes = [push_section(triangles, points, view) for triangles, _ in sections]
    save_figure(render.draw_meshes(sections), "bending_meshes")
    save_figure(render.draw_beams(sections, LOAD * -up_direction, beams), "bending_sections")
    save_figure(render.draw_pushes(sections, points, view, pushes), "bending_kern")
    save_animation(render.animate_turning(triangles, boundary, loads, turning), "bending_turning", DURATION_MS)

    # checks
    tips = np.array([render.flat(frames[-1] >> core.centroid(triangles)) - render.flat(core.centroid(triangles))
                     for (triangles, _), (_, frames, _, _) in zip(sections, beams)])   # [sections, 2]
    sideways, down = tips.T
    # Every tip moves down; the I-section's and the slotted tube's, mirror-symmetric about the load, straight
    # down; the turned I-section's and the Z-section's sideways as well.
    assert np.all(down < 0) and np.abs(sideways[[0, 3]]).max() < 1e-12
    assert np.abs(sideways[[1, 2]]).min() > 0.5 * np.abs(down[[1, 2]]).max()
    for (triangles, _), (strains, stress, _) in zip(sections, pushes):
        # The push inside the kern squeezes every fibre; the one outside stretches some.
        squeezed = stress.to_array()                                         # [pushes, triangles, 3]
        assert squeezed[0].max() < 0 and squeezed[1].max() > 0
        # A pull through one point stretches the fibre through another as much as the exchanged pull.
        rigidity = core.stiffness(triangles, MODULUS)
        first, second = core.pulls(rigidity, points[0]), core.pulls(rigidity, points[1])
        exchanged = (first & (core.beam_direction & points[1])) - (second & (core.beam_direction & points[0]))
        assert np.abs(exchanged.to_array()).max() < 1e-12


if __name__ == "__main__":
    main()
