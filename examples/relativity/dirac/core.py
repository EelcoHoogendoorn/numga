"""The Dirac electron in the spacetime algebra.

An electron's state at a point is an even multivector,
`psi = (PSEUDOSCALAR * (beta / 2)).exp() * rotor * np.sqrt(density)`: a density, an angle beta and a
Lorentz rotor. Sandwiching a vector with psi applies that rotor and scales by the density, so
`psi >> Vector` is a map on spacetime: it carries the observer's frame to the electron's.
Its image of the time axis is the electron's current, and its image of the z axis is its spin.
beta does not act on vectors at all. It acts on bivectors, where the spinor's own sandwich differs
from what its vector map extends to by the factor `(PSEUDOSCALAR * beta).exp() / density`.

For a plane wave the Dirac equation becomes a linear map on even multivectors, the Hamiltonian.
Its eigenvalues are plus and minus the energy, each fourfold. Right multiplication by the spin plane
`SPIN_PLANE` is a map that squares to minus one and commutes with the Hamiltonian. Positive-energy
states have `beta == 0` and negative-energy states `beta == np.pi`. A mixture of the two makes the
current circulate at twice the energy: the trembling motion, Zitterbewegung.

In matrix notation the spinor reads as a column of four complex numbers, the frame as the
bilinear covariants of the spinor with the four gamma matrices, the Hamiltonian as the alpha
matrices contracted with the momentum plus the Dirac beta matrix times the mass, and right
multiplication by the spin plane as multiplication by the imaginary unit.
"""

from __future__ import annotations

from itertools import accumulate

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import STA as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Vector = ga.gatype.vector()
Bivector = ga.gatype.bivector()
Even = ga.gatype.even()
# A Dirac spinor is an even multivector.
Spinor = Even
# The spinor's map on spacetime.
Frame = ga.gatype((Vector, Vector))                 # Vector <- Vector
# Vectors in the observer's space, perpendicular to the time axis.
Spatial = ga.gatype.from_blades("x y z")
# The plane-wave Dirac equation.
Hamiltonian = ga.gatype((Even, Even))               # Even <- Even
# The observer's time axis.
TIME = mv.t                                         # [] Vector
# The unit pseudoscalar, `mv.t ^ mv.x ^ mv.y ^ mv.z`.
PSEUDOSCALAR = mv.txyz                              # [] Pseudoscalar
# The spin plane `mv.y ^ mv.x`: right multiplication by it squares to minus one.
SPIN_PLANE = mv.yx                                  # [] Bivector


# --- math -----------------------------------------------------------------------------
def spinor(density: float, beta: float, rotor: Even) -> Spinor:
    """The spinor with the given density, angle beta and Lorentz rotor."""
    return (PSEUDOSCALAR * (beta / 2)).exp() * rotor * np.sqrt(density)


def frame(psi: Spinor) -> Frame:
    """The spinor's map on spacetime: the density times its Lorentz transformation. beta drops out,
    because the pseudoscalar anticommutes with vectors."""
    return psi >> Vector


def invariants(psi: Spinor) -> Even:
    """psi times its reverse: `(PSEUDOSCALAR * beta).exp() * density`, a scalar plus a pseudoscalar."""
    return psi.symmetric_reverse_product()


def duality(psi: Spinor):
    """What the spinor does to bivectors beyond what its frame does: its own sandwich on bivectors,
    composed with the inverse of the frame's extension to bivectors. It is multiplication by
    `(PSEUDOSCALAR * beta).exp() / density`."""
    return (psi >> Bivector)(frame(psi).outermorphism(Bivector).inverse())


def current(psi: Spinor) -> Vector:
    """The electron's current: the frame's image of the time axis."""
    return psi >> TIME


def spin(psi: Spinor) -> Vector:
    """The electron's spin, in units of half the reduced Planck constant: the frame's image of the z axis."""
    return psi >> mv.z


def velocity(flow: Vector) -> Bivector:
    """The velocity a current carries, as seen by the observer: its relative vector, the part
    across the time axis, over its density, the part along it."""
    return (flow ^ TIME) / (flow | TIME)


def hamiltonian(momentum: Vector, mass: float) -> Hamiltonian:
    """The Dirac equation for a plane wave of the given spatial momentum, as a map on spinors:
    the momentum as a relative vector on the left, and the mass through the reflection in the
    time axis."""
    return (momentum * TIME) * Even + mass * (TIME * Even * TIME)


def energy(momentum: Vector, mass: float) -> Scalar:
    """The energy of the plane wave: the root of the mass squared plus the momentum squared. A
    spatial vector squares to minus its length squared in this signature."""
    return (mass**2 - (momentum | momentum)).square_root()


def evolve(momentum: Vector, mass: float, psi: Spinor, times: np.ndarray) -> Spinor:
    """The plane wave's spinor at the given times.

    The projector onto positive energy, `0.5 * (psi + hamiltonian(momentum, mass)(psi) / E)`,
    splits the spinor into its positive- and negative-energy parts, which turn in the spin plane at
    the energy, in opposite senses.
    """
    E = energy(momentum, mass)
    positive = 0.5 * (psi + hamiltonian(momentum, mass)(psi) / E)
    turn = (SPIN_PLANE * E * times).exp()                                   # [times] Rotor
    return positive * turn.reverse() + (psi - positive) * turn


def path(velocities: Bivector, dt: float) -> Bivector:
    """The positions a velocity carries a point through, from the origin, one step at a time."""
    return stack(list(accumulate(velocities[:-1] * dt, initial=velocities[0] * 0.0)))
