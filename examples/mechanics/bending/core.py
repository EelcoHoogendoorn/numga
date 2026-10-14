"""Euler–Bernoulli beam bending, in the projective geometric algebra of three-dimensional space: a
cantilever loaded away from its principal bending directions bends sideways as well as along the load.

The beam runs along z, and its cross-section at the clamp lies in the plane `section_plane`, z = 0.
Bending keeps each cross-section plane: the motion from one cross-section to the next is a twist,
and unstrained it is the section's ideal line `shift_along_beam`, which carries the section along the
beam. Strain adds any other line in the section's plane, `Strain = wz yz zx`: a turn about a line in
the section, the neutral axis, or more shift along the beam. Each fibre is a line along the beam
through a point of the section; a strain stretches it by their pairing `strain & fibre`, and the fibre pulls along itself. Summed over
the section, the fibres pulling make the stiffness, a map from a strain to the resultant line of the
fibres' pull, `Stiffness = Resultant <- Strain`, with no centroid and no arms: the area, the first and
the second moments of area are all in the sum.

A load is a line. Where it meets the plane of a cross-section, joined with the point at infinity along
the beam, is the part of it the fibres there carry: a couple for a load across the beam, the load
itself for a push along it. The strain is one solve with the stiffness, and the running product of the
twists along the beam gives the frame of each cross-section. Each constant-strain piece is integrated
exactly, but loads and lever arms are evaluated on the straight reference beam: this is a
small-deflection load calculation with a finite-motion reconstruction, not nonlinear equilibrium.

The strain is itself the neutral axis, the line of fibres neither stretched nor squeezed. For a load
across the beam it runs through the centroid. For a section with a mirror symmetry along the load it
lies square to the load and the beam bends along it; for a Z-section, or an I-section turned in its
plane, it tilts, and the beam bends sideways as well as down. The stiffness is a polarity: a push
along the beam through a point of the section has as neutral axis that point's polar line, and the
points whose push compresses every fibre form the kern. The polarity is reciprocal: exchanging the
point of a pull and the fibre where stretch is measured gives the same response.

In the matrix notation of beam theory the curvatures read as
$\\begin{pmatrix}\\kappa_x\\\\ \\kappa_y\\end{pmatrix} = \\frac{1}{E}\\begin{pmatrix}I_{yy} & -I_{xy}\\\\ -I_{xy} & I_{xx}\\end{pmatrix}^{-1}\\begin{pmatrix}M_x\\\\ M_y\\end{pmatrix}$,
with $I_{xy}$ the product moment of area that a mirror symmetry would make zero.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext, concatenate
from numga.algebras import PGA3D as ga
from examples.mechanics.simplex import barycentric_samples, simplex_inertia_weights

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Point = ga.gatype.antivector()
Line = ga.gatype.bivector()
Motor = ga.gatype.rotor()
Plane = ga.gatype.vector()
# Lines in the section's plane.
Strain = ga.gatype(ga.subspace("wz yz zx"))
# Lines along the beam, and couples across it.
Resultant = ga.gatype(ga.subspace("xy wx wy"))
# A strain to the resultant of the fibres' pulls.
Stiffness = ga.gatype((Resultant, Strain))                 # Resultant <- Strain

# The cross-section at the clamp, the point at infinity along the beam, and the twist that carries a
# section along the beam.
section_plane = mv.z                                       # [] Plane
beam_direction = section_plane.dual()                      # [] Point
shift_along_beam = section_plane ^ mv.w                    # [] Line


# --- math -----------------------------------------------------------------------------
def areas(triangles: Point) -> Scalar:
    """Each triangle's area, from the plane through its three corners."""
    return (triangles[..., 0] & triangles[..., 1] & triangles[..., 2]).norm() / 2


def centroid(triangles: Point) -> Point:
    """The centroid of the whole section, one point for all its triangles: the mean of each triangle's corners,
    weighted by the triangle's area."""
    return (areas(triangles)[..., None] * triangles / 3).sum(axis=(-2, -1)).normalized()


def stiffness(triangles: Point, modulus: float) -> Stiffness:
    """Sum the fibres' pulls under a strain, integrating each triangle exactly."""
    weights = simplex_inertia_weights(3)
    samples = barycentric_samples(weights, triangles) * areas(triangles).square_root()[..., None]   # [triangles, samples] Point
    fibres = beam_direction & samples                                      # [triangles, samples] Resultant
    stretches = Strain & fibres                                             # [triangles, samples] Scalar <- Strain
    # Each point appears twice, so its square-root area supplies one area in this dyad.
    tensions = modulus * fibres * stretches                                 # [triangles, samples] Resultant <- Strain
    return tensions.sum(axis=(-2, -1))                                      # [] Resultant <- Strain


def carried(loads: Line, stations: np.ndarray) -> Resultant:
    """What the fibres carry at each station `[stations]` along the beam: the load seen from the
    station, met with the section's plane, and joined with the point at infinity along the beam as a
    fibre through that meet."""
    shifts = (shift_along_beam * (stations / 2)).exp()                       # [stations] Motor
    return beam_direction & ((shifts << loads[..., None]) ^ section_plane)   # [..., stations] Resultant


def bend(strains: Strain, step: float) -> Motor:
    """The frame of each cross-section from the clamp to the tip `[..., pieces + 1]`: the running
    product of the twists of the pieces `[..., pieces]` before it, each the shift along the beam plus
    the piece's strain."""
    twists = ((shift_along_beam + strains) * (step / 2)).exp()               # [..., pieces] Motor
    clamp = mv.rotor().broadcast_to(twists.shape[:-1] + (1,))                # [..., 1] Motor
    # Each twist acts in the frame of the cross-section before it, so each later twist multiplies
    # from the right: the reverse of the running product of the reverses, which multiplies from the left.
    return concatenate([clamp, twists.reverse().cumprod(axis=-1).reverse()], axis=-1)   # [..., pieces + 1] Motor


def stress(strains: Strain, modulus: float, points: Point) -> Scalar:
    """The stress along the beam at points of the section under the strains: the modulus times each
    fibre's stretch."""
    return modulus * (strains & (beam_direction & points))                   # [...] Scalar


def pulls(stiffness: Stiffness, points: Point) -> Strain:
    """The strain of a unit pull along the beam through each point of the section: its neutral axis,
    the point's polar line."""
    return stiffness.solve(beam_direction & points)                          # [...] Strain


# --- plumbing -------------------------------------------------------------------------
def point(coordinates: np.ndarray) -> Point:
    """Points of the section at coordinates `[..., 2]`: across and up."""
    return (mv("x y", coordinates) + mv.w).dual()

