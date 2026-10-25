"""Euler–Bernoulli beam bending with Saint-Venant torsion, in the projective geometric algebra of
three-dimensional space: thin-walled cantilevers loaded along any line, bending sideways when the load
is away from their principal bending directions, and twisting when it misses their shear centre.

The beam runs along z, and its cross-section at the clamp lies in the plane `section_plane`, z = 0.
The section is thin-walled: walls of a given thickness along a midline of straight pieces, each piece
a pair of points. Bending keeps each cross-section plane: the motion from one cross-section to the
next is a twist, and unstrained it is the section's ideal line `shift_along_beam`, which carries the
section along the beam. Bending adds any other line in the section's plane, `Strain = wz yz zx`: a
turn about a line in the section, the neutral axis, or more shift along the beam. Each fibre is a line
along the beam through a point of the section; a strain stretches it by their pairing `strain &
fibre`, and the fibre pulls along itself. Summed over the walls, the fibres pulling make the bending
stiffness, a map from a strain to the resultant line of the fibres' pull, `Bending = Resultant <- Strain`.

Twisting turns each cross-section about a line along the beam. Under a twist the walls do not stay
flat: each point of the midline moves along the beam in proportion to its warping, the area that the
line from a pole sweeps along the walls to it. The shear centre is the pole whose warping no bending
strain can see. The walls resist a twist by a third of their length times the cube of their thickness
each, and a slide of the section across the beam each along itself, by its area: the forces of the
slides pass through the shear centre, so a twist about the line along the beam through it costs no
slide, and the twist settles on that line.

A strain is any twist, a line of motion, and the whole stiffness, `Stiffness = Forque <- Twist`, takes it
to the forque, the line of force, of everything the section exerts: a sum of dyads of lines, the fibres'
for stretch and bending, the line at infinity of the section's plane for the twist, and the slides'
forces for shear. A load is a forque, and one solve gives the strain it causes. The running product of
the motors along the beam gives the frame of each cross-section. Each constant-strain piece is integrated
exactly, but loads and lever arms are evaluated on the straight reference beam, and the clamp is free to
warp: this is a small-deflection load calculation with a finite-motion reconstruction.

In the matrix notation of beam theory the curvatures read as
$\\begin{pmatrix}\\kappa_x\\\\ \\kappa_y\\end{pmatrix} = \\frac{1}{E}\\begin{pmatrix}I_{yy} & -I_{xy}\\\\ -I_{xy} & I_{xx}\\end{pmatrix}^{-1}\\begin{pmatrix}M_x\\\\ M_y\\end{pmatrix}$,
with $I_{xy}$ the product moment of area that a mirror symmetry would make zero, the rate of twist as
$\\theta' = T / (G J)$ with $J = \\sum b t^3 / 3$ and $T$ the torque about the shear centre, and the shear
strain as $\\gamma = V / (G A_s)$ with $A_s$ the shear area.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext, concatenate
from numga.algebras import PGA3D as ga
from examples.mechanics.simplex import barycentric_samples, simplex_inertia_weights

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Point = ga.gatype.antivector()
Twist = ga.gatype.bivector()
# Lines of force: loads, and the resultant of everything a section exerts.
Forque = ga.gatype.antibivector()
Motor = ga.gatype.rotor()
Plane = ga.gatype.vector()
# Lines in the section's plane.
Strain = Twist.derive.containing("z")
# Lines along the beam, and couples across it.
Resultant = Forque.derive.within("x y w")
# The fibres' pulls weighted by a warping, a swept area signed by the sense of the sweep: resultants of the
# other orientation, and their couples across the beam.
Warped = Resultant.derive.flipped_orientation
Couples = Warped.derive.containing("w")
# The shifts of a pole across the beam: points at infinity in the section's plane.
Shift = ga.gatype.antivector().derive.containing("w z")
# A bending strain to the resultant of the fibres' pulls.
Bending = ga.gatype((Resultant, Strain))                   # Resultant <- Strain
# Any strain to the forque of everything the section exerts.
Stiffness = ga.gatype((Forque, Twist))                      # Forque <- Twist

# The cross-section at the clamp, the point at infinity along the beam, and the twist that carries a
# section along the beam.
section_plane = mv.z                                       # [] Plane
beam_direction = section_plane.dual()                      # [] Point
shift_along_beam = section_plane ^ mv.w                    # [] Twist
# The couple about the beam: the line at infinity of the section's plane as a load, the join of the
# directions across and up.
couple_about_beam = mv.xy.dual()                           # [] Forque


# --- math -----------------------------------------------------------------------------
def lengths(walls: Point) -> Scalar:
    """Each wall's length, the size of the line through its two ends."""
    return (walls[..., 0] & walls[..., 1]).norm()


def samples(walls: Point, thickness: np.ndarray) -> Point:
    """Two points along each wall `[walls, 2]` that integrate any quadratic over the wall exactly, each
    carrying the square root of its share of the wall's area."""
    weights = simplex_inertia_weights(2)
    return barycentric_samples(weights, walls) * (lengths(walls) * thickness).square_root()[..., None]


def centroid(walls: Point, thickness: np.ndarray) -> Point:
    """The centroid of the whole section: the middle of each wall, weighted by the wall's area."""
    return ((walls[..., 0] + walls[..., 1]) * (lengths(walls) * thickness / 2)).sum(axis=-1).normalized()


def bending(walls: Point, thickness: np.ndarray, modulus: float) -> Bending:
    """Sum the fibres' pulls under a bending strain, integrating each wall exactly."""
    fibres = beam_direction & samples(walls, thickness)                     # [walls, 2] Resultant
    stretches = Strain & fibres                                             # [walls, 2] Scalar <- Strain
    # Each point appears twice, so its square-root area supplies one area in this dyad.
    tensions = modulus * fibres * stretches                                 # [walls, 2] Resultant <- Strain
    return tensions.sum(axis=(-2, -1))                                      # [] Resultant <- Strain


def warping(nodes: Point, ends: np.ndarray, thickness: np.ndarray, pole: Point) -> Warped:
    """The fibres weighted by their warping seen from the pole, summed over the walls: the pull of
    fibres stretched in proportion to the area that the line from the pole sweeps along the walls."""
    walls = nodes[ends]                                                     # [walls, 2] Point
    swept = beam_direction & pole & walls[..., 0] & walls[..., 1]           # [walls] Scalar
    warped = (swept * paths(ends, nodes.shape[0])).sum(axis=-1)            # [nodes] Scalar
    shares = (lengths(walls) * thickness).square_root()[..., None]          # [walls, 1] Scalar
    along = barycentric_samples(simplex_inertia_weights(2), warped[ends]) * shares   # [walls, 2] Scalar
    return (along * (beam_direction & samples(walls, thickness))).sum(axis=(-2, -1))  # [] Warped


def shear_centre(nodes: Point, ends: np.ndarray, thickness: np.ndarray) -> Point:
    """The pole whose warping no bending strain sees: its warped fibres pull with no couple across the
    beam. The warping is linear in the pole, so moving the pole from the centroid by a shift adds the
    warping seen from the shift itself, and one solve finds the shift that cancels the centroid's couples."""
    walls = nodes[ends]                                                     # [walls, 2] Point
    centre = centroid(walls, thickness)                                     # [] Point
    couples_per_shift = warping(nodes, ends, thickness, Shift).cast(Couples)   # [] Couples <- Shift
    centroid_couples = warping(nodes, ends, thickness, centre).cast(Couples)   # [] Couples
    return centre - couples_per_shift.solve(centroid_couples)                  # [] Point


def twisting(walls: Point, thickness: np.ndarray, shear_modulus: float) -> Scalar:
    """The torque per rate of twist of the open walls: a third of each wall's length times the cube of
    its thickness."""
    return shear_modulus * (lengths(walls) * thickness**3 / 3).sum(axis=-1)    # [] Scalar


def stiffness(nodes: Point, ends: np.ndarray, thickness: np.ndarray, modulus: float, shear_modulus: float
              ) -> Stiffness:
    """The whole stiffness of the section, a sum of dyads of lines: the fibres' pulls for stretch and
    bending; the line at infinity of the section's plane for the twist, which as a load is a couple
    about the beam; and for each wall, a force along it through the shear centre for a slide along it."""
    walls = nodes[ends]                                                     # [walls, 2] Point
    fibres = beam_direction & samples(walls, thickness)                     # [walls, 2] Resultant
    pulling = (modulus * fibres * (Twist & fibres)).sum(axis=(-2, -1))       # [] Resultant <- Twist
    twisted = twisting(walls, thickness, shear_modulus) * couple_about_beam * (couple_about_beam & Twist)   # [] Forque <- Twist
    along = (walls[..., 1] - walls[..., 0]) / lengths(walls)                # [walls] Point: each wall's direction
    slides = shear_centre(nodes, ends, thickness) & along                   # [walls] Forque
    sliding = (shear_modulus * lengths(walls) * thickness * slides * (slides & Twist)).sum(axis=-1)   # [] Forque <- Twist
    return pulling + twisted + sliding                                      # [] Forque <- Twist


def loads_at_stations(loads: Forque, stations: np.ndarray) -> Forque:
    """The loads seen from each station `[stations]` along the beam, as if the station were the clamp."""
    shifts = (shift_along_beam * (stations / 2)).exp()                       # [stations] Motor
    return shifts << loads[..., None]                                        # [..., stations] Forque


def bend(strains: Twist, step: float) -> Motor:
    """The frame of each cross-section from the clamp to the tip `[..., pieces + 1]`: the running
    product of the motors of the pieces `[..., pieces]` before it, each the shift along the beam plus
    the piece's strain."""
    piece_motors = ((shift_along_beam + strains) * (step / 2)).exp()               # [..., pieces] Motor
    clamp = mv.rotor().broadcast_to(piece_motors.shape[:-1] + (1,))                # [..., 1] Motor
    # Each twist acts in the frame of the cross-section before it, so each later twist multiplies
    # from the right: the reverse of the running product of the reverses, which multiplies from the left.
    return concatenate([clamp, piece_motors.reverse().cumprod(axis=-1).reverse()], axis=-1)   # [..., pieces + 1] Motor


def stress(strains: Twist, modulus: float, points: Point) -> Scalar:
    """The stress along the beam at points of the section under the strains: the modulus times each
    fibre's stretch."""
    return modulus * (strains & (beam_direction & points))                   # [...] Scalar


def pulls(bending: Bending, points: Point) -> Strain:
    """The strain of a unit pull along the beam through each point of the section: its neutral axis,
    the point's polar line."""
    return bending.solve(beam_direction & points)                            # [...] Strain


# --- plumbing -------------------------------------------------------------------------
def point(coordinates: np.ndarray) -> Point:
    """Points of the section at coordinates `[..., 2]`: across and up."""
    return (mv(Plane.derive.within("x y"), coordinates) + mv.w).dual()


def paths(ends: np.ndarray, count: int) -> np.ndarray:
    """For each node, the walls on the way to it `[nodes, walls]`, signed by their direction: the
    pseudo-inverse of the walls' incidence, which for the walls of an open section, a tree, sums each
    wall's step along the path to the node, up to one constant for all nodes."""
    incidence = np.eye(count)[ends[:, 1]] - np.eye(count)[ends[:, 0]]       # [walls, nodes]
    return np.linalg.pinv(incidence)
