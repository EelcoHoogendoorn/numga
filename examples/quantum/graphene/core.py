"""Electrons in graphene: the Dirac cones of the honeycomb lattice, the pseudospin that follows
the momentum around them, and the Berry phase of a loop around a cone, in the geometric algebra of
three-dimensional space.

An electron in graphene hops between neighbouring carbon atoms. Its state has two parts, one on
each of the honeycomb's two sublattices, and the hopping mixes them through a vector that depends
on the momentum, the pseudospin field `pseudospin(momentum, gap)`. Each of the three bonds
contributes the vector x turned in the plane by the phase `momentum | BONDS` across that bond; a
difference between the sublattices, as in boron nitride, adds a gap along z. The Hamiltonian is a
map on even multivectors, taking psi to `field * psi * mv.z`. Its eigenvalues are plus and minus
the length of the field, and the pseudospin of a state, its sandwich `psi >> mv.z`, points along
the field for the upper band and against it for the lower. At the corners of the Brillouin zone
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
Even = ga.gatype.even()
Rotor = ga.gatype.rotor()
Hamiltonian = ga.gatype((Even, Even))              # Even <- Even
# The energy of a hop between neighbouring atoms.
HOPPING = 2.8                                      # eV
# The bonds from an atom to its three neighbours, in carbon-carbon distances of 0.142 nm.
BONDS = mv.vector(np.array([[0.5, np.sqrt(3) / 2, 0.0], [0.5, -np.sqrt(3) / 2, 0.0], [-1.0, 0.0, 0.0]]))   # [3] Vector
# The two inequivalent corners of the Brillouin zone, where the cones sit.
VALLEYS = mv.vector(np.array([[2 * np.pi / 3, 2 * np.pi / (3 * np.sqrt(3)), 0.0],
                              [2 * np.pi / 3, -2 * np.pi / (3 * np.sqrt(3)), 0.0]]))            # [2] Vector


# --- math -----------------------------------------------------------------------------
def pseudospin(momentum: Vector, gap: Scalar) -> Vector:
    """The pseudospin field: minus the hopping times the sum over the bonds of x turned by the phase
    `momentum | BONDS` across each bond, plus the gap along z."""
    phases = momentum[..., None] | BONDS                                        # [..., 3] Scalar
    return -HOPPING * ((mv.xy * phases).exp() * mv.x).sum(axis=-1) + gap * mv.z   # [...] Vector


def hamiltonian(field: Vector) -> Hamiltonian:
    """The Hamiltonian as a map on even multivectors: the field on the left, z on the right."""
    return field * Even * mv.z                                                  # [...] Even <- Even


def direction(psi: Even) -> Vector:
    """The pseudospin direction of a state: its sandwich of z, over its norm."""
    return (psi >> mv.z) / psi.symmetric_reverse_product()                      # [...] Vector


def transport(directions: Vector) -> Rotor:
    """The rotors that carry a frame along a curve of unit directions, by the smallest rotation from
    each direction to the next: from the first direction to each of the others. Around a closed
    curve the last of them is the holonomy."""
    steps = (1 + directions[1:] * directions[:-1]).normalized()                 # [steps, ...] Rotor
    return steps.cumprod(axis=0)                                               # [steps, ...] Rotor


# --- plumbing -------------------------------------------------------------------------
def circle(count: int) -> Vector:
    """Unit vectors around the circle in the plane, in count steps, the last equal to the first."""
    angle = np.linspace(0.0, 2 * np.pi, count + 1)
    return mv.vector(np.stack([np.cos(angle), np.sin(angle), 0.0 * angle], axis=-1))   # [count + 1] Vector
