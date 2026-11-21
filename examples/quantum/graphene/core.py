"""Electrons in graphene: the Dirac cones of the honeycomb lattice, the pseudospin that follows
the momentum around them, and the Berry phase of a loop around a cone, in the geometric algebra of
three-dimensional space.

An electron in graphene hops between neighbouring carbon atoms. Its state has two parts, one on
each of the honeycomb's two sublattices, and the hopping mixes them through a field that depends
on the momentum, `pseudospin_field(momentum, gap)`, an extrinsic bivector like everything that
turns a spin. Each of the three bonds contributes the plane `yz` turned in the plane `xy` by the
phase `momentum & bonds` across that bond, the pairing of the momentum, a wavevector and so a
plane, with the bond, a displacement and so an antivector; a difference between the sublattices,
as in boron nitride, adds a gap in the plane `xy`. The Hamiltonian maps spinors, the even
multivectors, by `hamiltonian = -(Field * Spinor * xy)`. Its eigenvalues are plus and minus the
length of the field. A state's pseudospin, intrinsic like every spin, sandwiches the pseudospin
wholly on the first sublattice (`first_sublattice`, the patch of the plane `xy`) and divides by
the state's norm squared; it points along `field.dual()` for the upper band and against it for
the lower. At the corners of the Brillouin zone the field vanishes and the two bands meet in
cones.

Around a closed loop in momentum the pseudospin direction traces a closed curve on the unit
sphere. Carrying a frame along it by the smallest rotation from each direction to the next brings
the frame back turned about its starting direction. That holonomy is a rotor whose scalar part is
the cosine of the Berry phase, half the solid angle the curve encloses; around a gapless cone the
Berry phase is pi and the rotor is `-1`.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext, dim
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga, units=True).multivector
Scalar = ga.gatype.scalar()                                                     # Scalar+
# A momentum, read by pairing with a displacement, is a wavevector.
Wavevector = ga.gatype.vector()                                                 # Vector+
Displacement = ga.gatype.antivector()                                           # Antivector-
# The field that turns a pseudospin, extrinsic, and a pseudospin, intrinsic.
Field = ga.gatype.bivector()                                                    # Bivector+
Pseudospin = ga.gatype.antibivector()                                           # Antibivector-
Spinor = ga.gatype.even()                                                       # Even+
Rotor = ga.gatype.rotor()                                                       # Rotor+
# The energy of a hop between neighbouring atoms (eV) and the carbon-carbon bond length (0.142 nm).
hopping = mv.scalar(2.8, unit=dim.energy)                                       # [] Scalar+, energy
bond_length = mv.scalar(1.0, unit=dim.length)                                   # [] Scalar+, length
# The bonds from an atom to its three neighbours, a third of a turn apart: displacements, antivectors.
bond_angles = np.array([1, 3, 5]) * np.pi / 3
bond_turns = (mv.xy * (-bond_angles / 2)).exp()                                 # [3] Rotor+
bonds = (bond_turns >> mv.x.dual()) * bond_length                               # [3] Displacement-, length
# The two inequivalent corners of the Brillouin zone, where the cones sit.
valleys = mv.vector([[2 * np.pi / 3, 2 * np.pi / (3 * np.sqrt(3)), 0],
                     [2 * np.pi / 3, -2 * np.pi / (3 * np.sqrt(3)), 0]], unit=1 / dim.length)  # [2] Wavevector+, 1/length
# The pseudospin of a state wholly on the first sublattice: the patch of the plane xy.
first_sublattice = mv.xy.dual()                                                 # [] Pseudospin-
# The Hamiltonian, for a given field a map on spinors: minus the field on the left, the plane xy on the right.
hamiltonian = -(Field * Spinor * mv.xy)                                         # [] Spinor+ <- (Field+, Spinor+)


# --- math -----------------------------------------------------------------------------
def pseudospin_field(momentum: Wavevector, gap: Scalar) -> Field:
    """Hopping across the three bonds turns the plane `yz` in `xy` by each bond phase and adds the sublattice gap in `xy` (`[...] Field+`)."""
    phases = momentum[..., None] & bonds                                        # [..., 3] Scalar+
    return -hopping * ((mv.xy * phases).exp() * mv.yz).sum(axis=-1) + gap * mv.xy   # [...] Field+


def upper_pseudospins(offsets: Wavevector, gap: Scalar) -> Pseudospin:
    """Diagonalizing the Hamiltonian near both valleys gives the upper band's unit pseudospin (`[2, ...] Pseudospin-`)."""
    valley_field = pseudospin_field(valleys[:, None, None] + offsets, gap)      # [2, ...] Field+
    _, states = hamiltonian(valley_field).eigh()                                # [2, ..., 4] Spinor+
    upper_states = states[..., -1]                                              # [2, ...] Spinor+
    # The pseudospin of a state: its sandwich of the first sublattice's pseudospin, over its norm squared.
    return (upper_states >> first_sublattice) / upper_states.symmetric_reverse_product()   # [2, ...] Pseudospin-


def transport(directions: Pseudospin) -> Rotor:
    """Minimal step rotations along a curve of unit pseudospins accumulate into the frame transport rotors (`[...] Rotor+`), ending at the loop holonomy."""
    step_rotors = (1 + directions[1:] * directions[:-1]).normalized()           # [...] Rotor+
    return step_rotors.cumprod(axis=0)                                          # [...] Rotor+


# --- plumbing -------------------------------------------------------------------------
def momentum_circle(n_steps: int) -> Wavevector:
    """Steps a unit wavevector around the momentum circle in `n_steps` intervals (`[n_steps + 1] Wavevector+`), closing at the start."""
    angles = np.linspace(0.0, 2 * np.pi, n_steps + 1)
    turns = (mv.xy * (-angles / 2)).exp()                                       # [n_steps + 1] Rotor+
    return turns >> mv.x                                                        # [n_steps + 1] Wavevector+
