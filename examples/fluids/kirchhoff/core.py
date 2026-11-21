"""Kirchhoff's equations: a solid ellipsoid moving through still, ideal fluid, in the projective geometric algebra of three-dimensional space.

A body moving through water pushes the water aside, and the water it sets moving carries kinetic energy of its own. In an inviscid, incompressible fluid, that kinetic energy is a quadratic form in the body's rate of motion, so the surrounding fluid acts as anisotropic extra inertia carried along with the body, which is called its added mass. An elongated body moving end-on pushes little water aside while moving broadside displaces much more, so its momentum tilts away from its direction of motion and turns a body launched nearly end-on round toward broadside through the destabilizing Munk moment.

The ellipsoid is modelled by its shape map on directions, holding its squared semi-axes along its own axes, and the confocal ellipsoids form the pencil `shape + widening * Direction`. Averaged over that pencil and weighted by the body's volume over each confocal ellipsoid's volume, the inverse pencil gives the fluid's shift response and its outermorphism on couples gives the turn response. Added to the solid's inertia, both combine into one duality map from twists to forques whose pairing `rate & inertia(rate)` is twice the total kinetic energy of body and fluid. With no external force on either, the total impulse stays fixed in space, and at each instant the body moves at the twist `mobility(motor << impulse)` corresponding to that impulse in the body frame.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import NumpyContext, dim, stack
from numga.algebras import PGA3D as ga

mv = NumpyContext(ga, units=True).multivector
Scalar = ga.gatype.scalar()                                # Scalar+
Plane = ga.gatype.vector()                                 # Vector+
Point = ga.gatype.antivector()                             # Antivector-
# Offsets and point velocities are points at infinity: degenerate antivectors with intrinsic orientation.
Direction = Point.derive.degenerate                        # Antivector-
Twist = ga.gatype.bivector()                               # Bivector+
Forque = ga.gatype.antibivector()                          # Antibivector-
# Couples are forques along a line at infinity: the angular momentum of a pure turn.
Couple = Forque.derive.degenerate                          # Antibivector-
Motor = ga.gatype.rotor()                                  # Rotor+
# A stretch map carrying the unit sphere onto the ellipsoid.
Stretch = ga.gatype((Direction, Direction))                # Direction- <- Direction-
# A shape map holding the squared semi-axes along the body's axes.
Shape = ga.gatype((Direction, Direction))                  # Direction- <- Direction-
# An inertia duality map sending twists to forques.
Inertia = ga.gatype((Forque, Twist))                       # Forque- <- Twist+

# The body's centre, the coordinate planes through it, and the directions normal to those planes.
centre = mv.point([0, 0, 0], unit=dim.length)              # [] Point-, 1 | length
axes = stack([mv.x, mv.y, mv.z])                           # [3] Plane+
directions = axes.dual()                                   # [3] Direction-
# The velocity of the centre under a twist, and the couple extracted from its rotational part.
centre_velocity = centre.commutator(Twist)                 # [] Direction- <- Twist+
turn_couple = directions.regressive(directions.commutator(Twist)).sum(axis=-1) / 2  # [] Couple- <- Twist+


# --- math -----------------------------------------------------------------------------
def ellipsoid_stretch(semi_axes: Scalar) -> Stretch:
    """Linear stretch carrying the unit sphere onto an ellipsoid with the given semi-axes (`[..., 3] Scalar+, length`) along the body axes."""
    # Scale each direction component along its corresponding axis plane by that axis's semi-axis length.
    return (directions * semi_axes * (axes & Direction)).sum(axis=-1)   # [...] Direction- <- Direction-, length


def total_inertia(shape: Shape, density: Scalar, fluid_density: Scalar, n_widenings: int) -> Inertia:
    """Combined solid and added-fluid inertia of an ellipsoid in an ideal fluid."""
    widenings, weights = quadrature(n_widenings)
    # Scale quadrature widenings by the mean squared semi-axis, and obtain the ellipsoid volume from det(shape).
    mean_square = shape.trace() / len(axes)                # [...] Scalar+, length^2
    volume = 4 / 3 * np.pi * shape.det().square_root()     # [...] Scalar+, length^3
    # Confocal ellipsoids grow every squared semi-axis equally, weighted by the ratio of body volume to confocal volume.
    confocal_shapes = shape[..., None] + (mean_square[..., None] * widenings) * Direction   # [..., n_widenings] Direction- <- Direction-, length^2
    quadrature_weights = (shape.det()[..., None] / confocal_shapes.det()).square_root() * mean_square[..., None] * weights   # [..., n_widenings] Scalar+, length^2
    # Confocal averages of the inverse shape and its couple outermorphism give the shift and turn depolarization factors.
    shift_factors = (confocal_shapes.inverse() * quadrature_weights).sum(axis=-1)   # [...] Direction- <- Direction-
    turn_factors = (confocal_shapes.inverse().outermorphism(Couple) * quadrature_weights).sum(axis=-1)   # [...] Couple- <- Couple-, 1/length^2
    # On each coordinate plane, `areas` multiplies the two squared semi-axes and `axis_sum` adds them via the mixed
    # outermorphism term; their discriminant `(a^2 - b^2)^2` vanishes for turns about an axis of symmetry.
    areas = shape.outermorphism(Couple)                    # [...] Couple- <- Couple-, length^4
    axis_sum = (mean_square * Direction + shape).outermorphism(Couple) / mean_square - mean_square * Couple - areas / mean_square   # [...] Couple- <- Couple-, length^2
    discriminant = axis_sum(axis_sum) - 4 * areas          # [...] Couple- <- Couple-, length^4
    # Dimensionless translational and rotational added-mass operators per unit displaced fluid mass.
    fluid_shift = (2 * Direction - shift_factors).solve(shift_factors)   # [...] Direction- <- Direction-
    fluid_turn = (2 * Couple - axis_sum(turn_factors)).solve(discriminant(turn_factors)) / 5   # [...] Couple- <- Couple-, length^2
    # Combine the solid's isotropic mass and fifth-moment rotational inertia with the displaced fluid's added inertia.
    shift_inertia = (volume * density) * Direction + (volume * fluid_density) * fluid_shift   # [...] Direction- <- Direction-, mass
    turn_inertia = (volume * density / 5) * axis_sum + (volume * fluid_density) * fluid_turn   # [...] Couple- <- Couple-, mass*length^2
    # Translational momentum is a line through the centre; rotational momentum is an ideal couple.
    return (centre & shift_inertia(centre_velocity)) + turn_inertia(turn_couple)   # [...] Forque- <- Twist+


def trajectory(motor: Motor, impulse: Forque, inertia: Inertia, dt: Scalar, n_steps: int) -> Iterator[Motor]:
    """Yield the body's motor placement from the start and after each Lie midpoint step under a space-fixed impulse."""
    mobility = inertia.inverse()                           # [...] Twist+ <- Forque-
    yield motor
    for _ in range(n_steps):
        midpoint = motor * (mobility(motor << impulse) * (-dt / 4)).exp()   # [...] Motor+
        motor = motor * (mobility(midpoint << impulse) * (-dt / 2)).exp()   # [...] Motor+
        yield motor


# --- plumbing -------------------------------------------------------------------------
def quadrature(n_widenings: int) -> tuple[np.ndarray, np.ndarray]:
    """Confocal widening nodes on the positive half-line and their quadrature weights via Gauss-Legendre mapping."""
    nodes, weights = np.polynomial.legendre.leggauss(n_widenings)
    unit_nodes = (nodes + 1) / 2
    return 1 / unit_nodes**2 - 1, weights / unit_nodes**3
