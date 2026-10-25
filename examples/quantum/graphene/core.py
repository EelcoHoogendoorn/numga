"""Electrons in graphene: the Dirac cones of the honeycomb lattice, the pseudospin that follows
the momentum around them, and the Berry phase of a loop around a cone, in the geometric algebra of
three-dimensional space.

An electron in graphene hops between neighbouring carbon atoms. Its state has two parts, one on
each of the honeycomb's two sublattices, and the hopping mixes them through a field that depends
on the momentum, the pseudospin field `pseudospin_field(momentum, gap)`, an extrinsic bivector like
everything that turns a spin. Each of the three bonds contributes the plane yz turned in the plane xy
by the phase `momentum & BONDS` across that bond, the pairing of the momentum, a wavevector and so a
plane, with the bond, a displacement and so an antivector; a difference between the sublattices, as in
boron nitride, adds a gap in the plane xy. The Hamiltonian is a map on even multivectors, taking psi
to `-(field * psi * mv.xy)`. Its eigenvalues are plus and minus the length of the field. The
pseudospin of a state, intrinsic like every spin, is its sandwich of the pseudospin wholly on the
first sublattice, `psi >> FIRST_SUBLATTICE`, the patch of the plane xy; it lies along the field,
`field.dual()`, for the upper band and against it for the lower. At the corners of the Brillouin zone
the field vanishes and the two bands meet in cones.

Around a closed loop in momentum the pseudospin direction traces a closed curve on the unit sphere.
Carrying a frame along it, by the smallest rotation from each direction to the next, brings it
back turned about its starting direction. That holonomy is a rotor with scalar part
`np.cos(gamma)`: the frame has turned by `2 * gamma`, the solid angle the curve encloses, and
gamma is the Berry phase. Around a gapless cone it is pi, and the rotor is -1.

In matrix notation the state reads as a column of two complex amplitudes, the Hamiltonian as the
2x2 matrix that contracts the pseudospin field with the Pauli matrices, and the Berry phase as the
phase of the product of the overlaps of neighbouring eigenvectors around the loop.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Vector = ga.gatype.vector()
# The field that turns a pseudospin, extrinsic, and a pseudospin, intrinsic.
Field = ga.gatype.bivector()
Pseudospin = Vector.derive.flipped_orientation
Even = ga.gatype.even()
Rotor = ga.gatype.rotor()
Hamiltonian = ga.gatype((Even, Even))              # Even <- Even
# The energy of a hop between neighbouring atoms.
HOPPING = 2.8                                      # eV
# The bonds from an atom to its three neighbours, in carbon-carbon distances of 0.142 nm: displacements,
# antivectors with intrinsic orientation.
BONDS = mv.vector([[0.5, np.sqrt(3) / 2, 0.0], [0.5, -np.sqrt(3) / 2, 0.0], [-1.0, 0.0, 0.0]]).dual()   # [3] Point
# The two inequivalent corners of the Brillouin zone, where the cones sit.
VALLEYS = mv.vector([[2 * np.pi / 3, 2 * np.pi / (3 * np.sqrt(3)), 0.0],
                     [2 * np.pi / 3, -2 * np.pi / (3 * np.sqrt(3)), 0.0]])                     # [2] Vector
# The pseudospin of a state wholly on the first sublattice: the patch of the plane xy.
FIRST_SUBLATTICE = mv.x.dual() & mv.y.dual()                                   # [] Pseudospin


# --- math -----------------------------------------------------------------------------
def pseudospin_field(momentum: Vector, gap: np.ndarray) -> Field:
    """The pseudospin field: minus the hopping times the sum over the bonds of the plane yz turned in the
    plane xy by the phase `momentum & BONDS` across each bond, plus the gap in the plane xy."""
    phases = momentum[..., None] & BONDS                                        # [..., 3] Scalar
    return -HOPPING * ((mv.xy * phases).exp() * mv.yz).sum(axis=-1) + gap * mv.xy   # [...] Field


def hamiltonian(field: Field) -> Hamiltonian:
    """The Hamiltonian as a map on even multivectors: minus the field on the left, the plane xy on the right."""
    return -(field * Even * mv.xy)                                              # [...] Even <- Even


def direction(psi: Even) -> Pseudospin:
    """The pseudospin direction of a state: its sandwich of the first sublattice's pseudospin, over its norm."""
    return (psi >> FIRST_SUBLATTICE) / psi.symmetric_reverse_product()          # [...] Pseudospin


def transport(directions: Pseudospin) -> Rotor:
    """The rotors that carry a frame along a curve of unit directions, by the smallest rotation from
    each direction to the next: from the first direction to each of the others. Around a closed
    curve the last of them is the holonomy."""
    steps = (1 + directions[1:] * directions[:-1]).normalized()                 # [steps, ...] Rotor
    return steps.cumprod(axis=0)                                               # [steps, ...] Rotor


# --- plumbing -------------------------------------------------------------------------
def circle(count: int) -> Vector:
    """Unit vectors around the circle in the plane, in count steps, the last equal to the first."""
    angle = np.linspace(0.0, 2 * np.pi, count + 1)
    return (mv.xy * (-angle / 2)).exp() >> mv.x                        # [count + 1] Vector
