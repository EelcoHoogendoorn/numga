"""Electrons in graphene: the Dirac cones of the honeycomb lattice, the pseudospin that follows
the momentum around them, and the Berry phase of a loop around a cone, in the geometric algebra of
three-dimensional space.

An electron in graphene hops between neighbouring carbon atoms. Its state has two parts, one on
each of the honeycomb's two sublattices, and the hopping mixes them through a field that depends
on the momentum, the pseudospin field `pseudospin_field(momentum, gap)`, an extrinsic bivector like
everything that turns a spin. Each of the three bonds contributes the plane yz turned in the plane xy
by the phase `momentum & bonds` across that bond, the pairing of the momentum, a wavevector and so a
plane, with the bond, a displacement and so an antivector; a difference between the sublattices, as in
boron nitride, adds a gap in the plane xy. The Hamiltonian is a map on spinors, the even multivectors,
`hamiltonian(field)` taking psi to `-(field * psi * mv.xy)`. Its eigenvalues are plus and minus the
length of the field. The pseudospin of a state, intrinsic like every spin, is its sandwich of the
pseudospin wholly on the first sublattice, `psi >> first_sublattice`, the patch of the plane xy; it
lies along the field, `field.dual()`, for the upper band and against it for the lower. At the corners
of the Brillouin zone the field vanishes and the two bands meet in cones.

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
Scalar = ga.gatype.scalar()                                                     # Scalar+
# A momentum, read by pairing with a displacement, is a wavevector.
Wavevector = ga.gatype.vector()                                                 # Vector+
# The field that turns a pseudospin, extrinsic, and a pseudospin, intrinsic.
Field = ga.gatype.bivector()                                                    # Bivector+
Pseudospin = ga.gatype.antibivector()                                           # Antibivector-
Spinor = ga.gatype.even()                                                       # Even+
Rotor = ga.gatype.rotor()                                                       # Rotor+
# The energy of a hop between neighbouring atoms, in eV.
HOPPING = 2.8
# The bonds from an atom to its three neighbours, a third of a turn apart, in carbon-carbon distances of 0.142 nm:
# displacements, antivectors with intrinsic orientation.
BOND_ANGLES = np.array([1, 3, 5]) * np.pi / 3
bonds = (mv.xy * (-BOND_ANGLES / 2)).exp() >> mv.x.dual()                      # [bonds] Displacement-
# The two inequivalent corners of the Brillouin zone, where the cones sit.
valleys = mv.vector([[2 * np.pi / 3, 2 * np.pi / (3 * np.sqrt(3)), 0],
                     [2 * np.pi / 3, -2 * np.pi / (3 * np.sqrt(3)), 0]])        # [valleys] Wavevector+
# The pseudospin of a state wholly on the first sublattice: the patch of the plane xy.
first_sublattice = mv.xy.dual()                                                 # [] Pseudospin-
# The Hamiltonian, for a given field a map on spinors: minus the field on the left, the plane xy on the right.
hamiltonian = -(Field * Spinor * mv.xy)                                         # [] Spinor+ <- (Field+, Spinor+)


# --- math -----------------------------------------------------------------------------
def pseudospin_field(momentum: Wavevector, gap: np.ndarray) -> Field:
    """The pseudospin field: minus the hopping times the sum over the bonds of the plane yz turned in the
    plane xy by the phase `momentum & bonds` across each bond, plus the gap in the plane xy."""
    phases = momentum[..., None] & bonds                                        # [..., bonds] Scalar+
    return -HOPPING * ((mv.xy * phases).exp() * mv.yz).sum(axis=-1) + gap * mv.xy   # [...] Field+


def transport(directions: Pseudospin) -> Rotor:
    """The rotors that carry a frame along a curve of unit directions, by the smallest rotation from
    each direction to the next: from the first direction to each of the others. Around a closed
    curve the last of them is the holonomy."""
    steps = (1 + directions[1:] * directions[:-1]).normalized()                 # [steps, ...] Rotor+
    return steps.cumprod(axis=0)                                                # [steps, ...] Rotor+


# --- plumbing -------------------------------------------------------------------------
def circle(n_steps: int) -> Wavevector:
    """Unit wavevectors around the circle in the plane, in n_steps steps, the last equal to the first."""
    angle = np.linspace(0.0, 2 * np.pi, n_steps + 1)
    return (mv.xy * (-angle / 2)).exp() >> mv.x                                 # [steps + 1] Wavevector+
