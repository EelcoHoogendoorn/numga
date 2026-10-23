"""Scenes for thin-walled beam bending and twisting: an I-section, a turned I-section, a Z-section and a
slotted tube, each clamped at one end and loaded straight down at the other through its centroid; the
Z-section with its load turning once around; the slotted tube with its load moved across, through its
shear centre; a stubby I-beam, with its walls' shear stiffness and with its slides held rigid; and each
section pushed along the beam off its centroid, once inside its kern and once outside."""

from __future__ import annotations

import numpy as np

from examples.mechanics.bending import core
from examples.mechanics.bending.mesh import NAMES, section_walls

mv, Strain = core.mv, core.Strain
# The point at infinity up the section.
up_direction = mv.y.dual()                                                   # [] Point
# The material's modulus and Poisson's ratio, the beam's length, the pieces it is cut into, and the load at
# its tip.
MODULUS = 1.0
POISSON = 0.3
SHEAR_MODULUS = MODULUS / (2 * (1 + POISSON))
LENGTH = 10.0
N_PIECES = 40
LOAD = 1e-5
# The frames of the load's turn and of its move across, and the duration of each.
N_FRAMES = 48
DURATION_MS = 60
# The ends of the slotted tube's load's move across, across and up.
ACROSS = [[0.4, 0.0], [-1.15, 0.0]]
# The stubby beam's length and load, and a shear modulus a million times the material's: the slides held
# practically rigid, as Euler–Bernoulli theory takes them.
STUBBY_LENGTH = 2.0
STUBBY_LOAD = 1.5e-3
RIGID_SHEAR = 1e6 * SHEAR_MODULUS
# The pushes along the beam: their size, and the points they go through, across and up, one inside
# every section's kern and one outside.
PUSH = 1e-3
PUSH_POINTS = [[0.03, 0.04], [0.1, 0.3]]


# --- math -----------------------------------------------------------------------------
def tip(points: core.Point, length: float) -> core.Point:
    """Points of the section carried to the tip of the unbent beam."""
    return (core.shift_along_beam * (length / 2)).exp() >> points


def cantilever(nodes: core.Point, ends: np.ndarray, thickness: np.ndarray, loads: core.Forque, length: float,
               shear_modulus: float) -> tuple[core.Twist, core.Motor, core.Strain, core.Scalar]:
    """A cantilever of a length, under load lines at its tip, batched over loads: the strain of each piece
    `[..., pieces]`, the frame of each cross-section `[..., pieces + 1]`, the principal bending axes `[2]`,
    and the stress at the clamp at the ends of the section's walls `[..., walls, 2]`."""
    step = length / N_PIECES
    # The middle of each piece.
    stations = (np.arange(N_PIECES) + 0.5) * step                             # [pieces]
    walls = nodes[ends]                                                     # [walls, 2] Point
    rigidity = core.bending(walls, thickness, MODULUS)                      # [] Resultant <- Strain
    whole = core.stiffness(nodes, ends, thickness, MODULUS, shear_modulus)  # [] Forque <- Twist
    # Lever arms are measured on the straight reference beam. A strain moves the frames by
    # `exp(strain * step / 2)`, against its own sense, so it is solved against the load turned in sign.
    strains = whole.solve(-core.loads_at_stations(loads, stations))         # [..., pieces] Twist
    frames = core.bend(strains, step)                                       # [..., pieces + 1] Motor
    # Principal bending axes pass through the centroid; the metric does not see the shift along the
    # beam. Weakest first, then the shift.
    _, axes = (Strain | Strain).eigh(Strain & rigidity)                    # [3] Strain
    stress = core.stress(strains[..., 0, None, None], MODULUS, walls)      # [..., walls, 2] Scalar
    return strains, frames, axes[:2], stress


def turning_loads() -> core.Point:
    """The tip load turning once around the beam, in the plane of the section."""
    angles = np.linspace(0, 2 * np.pi, N_FRAMES, endpoint=False)               # [frames]
    return (mv.xy * (-angles / 2)).exp() >> (LOAD * -up_direction)           # [frames] Point


def across_points() -> core.Point:
    """The points the slotted tube's load goes through as it moves across the section."""
    start, end = core.point(ACROSS)                                          # [] Point each
    fractions = np.linspace(0, 1, N_FRAMES)                                    # [frames]
    return start * (1 - fractions) + end * fractions                        # [frames] Point


def push_section(nodes: core.Point, ends: np.ndarray, thickness: np.ndarray, points: core.Point
                 ) -> tuple[core.Strain, core.Scalar, core.Strain]:
    """Pushes along the beam through points of a section: their strains `[pushes]` and stresses at the
    ends of the section's walls `[pushes, walls, 2]`; and the polar of every node `[nodes]`, the neutral
    axis of a pull through it. By reciprocity a pull through a point on a node's polar leaves that node's
    fibre unstretched, so the polars bound the kern, the points whose push squeezes every fibre."""
    walls = nodes[ends]                                                     # [walls, 2] Point
    rigidity = core.bending(walls, thickness, MODULUS)                      # [] Resultant <- Strain
    strains = -PUSH * core.pulls(rigidity, points)                          # [pushes] Strain
    stress = core.stress(strains[..., None, None], MODULUS, walls)           # [pushes, walls, 2] Scalar
    return strains, stress, core.pulls(rigidity, nodes)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.mechanics.bending import render

    sections = section_walls()
    down = LOAD * -up_direction                                              # [] Point
    centres = [core.centroid(nodes[ends], thickness) for nodes, ends, thickness in sections]
    beams = [cantilever(*section, tip(centre, LENGTH) & down, LENGTH, SHEAR_MODULUS) for section, centre in zip(sections, centres)]
    zed = sections[NAMES.index("Z-section")]
    turning = turning_loads()
    turned = cantilever(*zed, tip(centres[NAMES.index("Z-section")], LENGTH) & turning, LENGTH, SHEAR_MODULUS)
    tube = sections[NAMES.index("slotted tube")]
    across = across_points()
    moved = cantilever(*tube, tip(across, LENGTH) & down, LENGTH, SHEAR_MODULUS)
    stubby = sections[NAMES.index("I-section")]
    stubby_load = tip(centres[NAMES.index("I-section")], STUBBY_LENGTH) & (STUBBY_LOAD * -up_direction)   # [] Forque
    stubs = [cantilever(*stubby, stubby_load, STUBBY_LENGTH, shear) for shear in (SHEAR_MODULUS, RIGID_SHEAR)]
    points = core.point(PUSH_POINTS)                                         # [pushes] Point
    pushes = [push_section(*section, points) for section in sections]
    shear_centres = [core.shear_centre(*section) for section in sections]
    save_figure(render.draw_walls(sections, centres, shear_centres), "bending_walls")
    save_figure(render.draw_beams(sections, centres, shear_centres, down, beams), "bending_sections")
    save_figure(render.draw_pushes(sections, centres, points, pushes), "bending_kern")
    save_animation(render.animate_loads(zed, centres[NAMES.index("Z-section")], shear_centres[NAMES.index("Z-section")],
                                        centres[NAMES.index("Z-section")], turning, turned), "bending_turning", DURATION_MS)
    save_animation(render.animate_loads(tube, centres[NAMES.index("slotted tube")], shear_centres[NAMES.index("slotted tube")],
                                        across, down, moved), "bending_across", DURATION_MS)
    save_figure(render.draw_stubby(stubby, [frames for _, frames, _, _ in stubs], centres[NAMES.index("I-section")],
                                   STUBBY_LOAD * -up_direction), "bending_stubby")

    # checks
    tips = np.array([render.flat(frames[-1] >> centre) - render.flat(centre)
                     for centre, (_, frames, _, _) in zip(centres, beams)])   # [sections, 2]
    sideways, falls = tips.T
    # Every tip moves down; the I-section's, mirror-symmetric about the load, straight down; the turned
    # I-section's and the Z-section's sideways as well.
    assert np.all(falls < 0) and np.abs(sideways[0]) < 1e-12
    assert np.abs(sideways[[1, 2]]).min() > 0.5 * np.abs(falls[[1, 2]]).max()
    # The slotted tube's load through its centroid passes beside its shear centre and twists it; moved
    # across, the load twists it one way, then the other, and through the shear centre not at all.
    # The turn about the beam, per unit length.
    twist_rates = (moved[0][:, 0] & core.shift_along_beam).to_array()             # [frames]
    assert twist_rates[0] * twist_rates[-1] < 0
    passing = np.argmin(np.abs(render.flat(across)[:, 0] - render.flat(shear_centres[-1])[0]))
    assert np.abs(twist_rates[passing]) < 0.1 * np.abs(twist_rates).max()
    # The stubby beam's slides add a deflection comparable to its bending's.
    sliding, rigid = (render.flat(frames[-1] >> centres[0])[1] for _, frames, _, _ in stubs)
    assert sliding < rigid < 0 and sliding < 1.2 * rigid
    for (nodes, ends, thickness), (strains, stress, polars) in zip(sections, pushes):
        # The push inside the kern squeezes every fibre; the one outside stretches some.
        squeezed = stress.to_array()                                         # [pushes, walls, 2]
        assert squeezed[0].max() < 0 and squeezed[1].max() > 0
        # A pull through one point stretches the fibre through another as much as the exchanged pull.
        rigidity = core.bending(nodes[ends], thickness, MODULUS)
        first, second = core.pulls(rigidity, points[0]), core.pulls(rigidity, points[1])
        exchanged = (first & (core.beam_direction & points[1])) - (second & (core.beam_direction & points[0]))
        assert np.abs(exchanged.to_array()).max() < 1e-12


if __name__ == "__main__":
    main()
