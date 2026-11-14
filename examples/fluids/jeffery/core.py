"""Jeffery orbits in the geometric algebra of three-dimensional space.

A slow, viscous shear flow moves each offset at a velocity linear in the offset, modelled as a map `Gradient = Point <- Point`. A drop of the fluid turns and stretches with the flow, whereas a rigid particle can only turn, settling at the unique spin rate that leaves zero net torque on its surface, found by Jeffery in 1922.

Each particle is modelled as an ellipsoid whose dual quadric `Shape = Point <- Plane` carries the unit sphere's polarity `Plane.dual()` by the body's stretch, sending offsets forward and pulling planes back by its adjugate, `stretch(Plane.dual()(stretch.adjugate()))`, with surface offsets satisfying `shape.solve(offset) & offset == 1`. Jeffery's spin, a bivector, satisfies one equation between maps:

    Point.commutator(spin)(shape) + shape(Plane.commutator(spin)) == gradient(shape) - shape(gradient.adjugate())

On the left is the dual quadric's rate of change as the body turns at `spin`; on the right is the flow gradient acting on the shape's offsets less its adjugate acting on the planes. Because both maps are antisymmetric, joining their output with their plane's free offset via `(Point & map(Plane)).trace(1, 2)` captures all three independent components without a metric, reducing torque balance at any orientation to a single linear solve. A sphere spins at half the curl of the gradient, a rod traces a closed Jeffery orbit on the sphere of directions, and a triaxial ellipsoid precesses quasiperiodically.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import NumpyContext
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
# Offsets and velocities are antivectors with intrinsic orientation.
Point = ga.gatype.antivector()                             # Antivector-
Rate = ga.gatype.bivector()                                # Bivector+
Rotor = ga.gatype.rotor()                                  # Rotor+
# Planes that read offsets are vectors with extrinsic orientation.
Plane = ga.gatype.vector()                                 # Vector+
# Linear stretch carrying the unit sphere onto the body.
Stretch = ga.gatype((Point, Point))                        # Point- <- Point-
# Dual quadric of the body.
Shape = ga.gatype((Point, Plane))                          # Point- <- Plane+
# Velocity gradient of the ambient flow.
Gradient = ga.gatype((Point, Point))                       # Point- <- Point-

# Coordinate planes across the flow, across the gradient, and parallel to the shear plane.
axes = mv.basis()                                          # [3] Plane+


# --- math -----------------------------------------------------------------------------
def body_stretches(semi_axes: np.ndarray) -> Stretch:
    """Linear stretches carrying the unit sphere onto ellipsoids with the given principal semi-axes."""
    # Scale each offset along each coordinate axis by its height above that axis's plane.
    return (axes.dual() * semi_axes * (axes & Point)).sum(axis=-1)   # [...] Point- <- Point-


def flow_curl(gradient: Gradient) -> Rate:
    """Vorticity bivector whose commutator action equals the velocity gradient minus its metric adjoint."""
    # The reciprocal of an orthonormal antivector basis is its reverse; the commutator spans the turning plane.
    return Point.reverse().commutator(gradient(Point)).contract(1, 2)   # [...] Rate+


def jeffery_spin(shape: Shape, gradient: Gradient) -> Rate:
    """Torque-free turning rate of a rigid ellipsoid of the given dual quadric in a linear velocity field."""
    # Rate of change of the dual quadric under rigid turning of both its output offsets and input planes.
    turning_rate = Point.commutator(Rate)(shape) + shape(Plane.commutator(Rate))   # [...] Point- <- (Plane+, Rate+)
    # Skew action of the ambient velocity gradient on offsets against its adjugate on planes.
    flow_rate = gradient(shape) - shape(gradient.adjugate())   # [...] Point- <- Plane+
    # Contracting each skew map with the regressive join isolates its three independent bivector-like components.
    return (Point & turning_rate(Plane)).trace(1, 2).solve((Point & flow_rate(Plane)).trace(1, 2))   # [...] Rate+


def body_shapes(stretches: Stretch, rotors: Rotor) -> Shape:
    """Dual quadrics of ellipsoids with principal stretches turned by their orientation rotors."""
    turned_stretch = rotors >> stretches(rotors << Point)      # [...] Point- <- Point-
    # Metric roundness enters once through the unit sphere's duality map from planes to normal offsets.
    unit_sphere = Plane.dual()                                 # [] Point- <- Plane+
    return turned_stretch(unit_sphere(turned_stretch.adjugate()))   # [...] Point- <- Plane+


def midpoint_step(rotors: Rotor, stretches: Stretch, gradient: Gradient, dt: float) -> Rotor:
    """Orientation rotors advanced across a timestep `dt` by the midpoint Jeffery spin."""
    halfway_rotors = (jeffery_spin(body_shapes(stretches, rotors), gradient) * (-dt / 4)).exp() * rotors   # [...] Rotor+
    return (jeffery_spin(body_shapes(stretches, halfway_rotors), gradient) * (-dt / 2)).exp() * rotors     # [...] Rotor+


def rotor_trajectory(rotors: Rotor, stretches: Stretch, gradient: Gradient, dt: float, n_steps: int) -> Iterator[Rotor]:
    """Yield the initial orientation rotors followed by `n_steps` midpoint-integrated steps."""
    yield rotors
    for _ in range(n_steps):
        rotors = midpoint_step(rotors, stretches, gradient, dt)
        yield rotors


