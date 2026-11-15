"""A tricycle steered by its front wheel drives in the plane and in space under the same equations.

Each axle is modelled as a plane of projective geometric algebra, a hyperplane that stands vertical in space. The rear axle lies across the origin, and the front axle at rest lies a wheelbase ahead along x. Steering turns the front axle about the steering axis, where the front axle at rest meets the symmetry plane y = 0 down the middle of the frame, which is a point in the plane and a vertical line in space.

Where the rear and front axles meet is the turning centre, `rear ^ front`, a point in the plane and a vertical line in space. Every wheel rolls around that meet because every axle passes through it, and the tricycle moves by its exponential: a rotation about the turning centre, or a pure translation when the front wheel is straight and the parallel axles meet at infinity. Because the meet's weight equals the sine of the steering angle and its distance from the front wheel equals the wheelbase divided by that sine, a constant step multiplier rolls the front wheel by the same distance at every steering angle.

Across a drive the steering angle varies from step to step. Each step moves the tricycle in its body frame by the exponential of that step's turning centre, so step motors multiply on the right of the pose; reversing the cumulative left product of their reverses accumulates that right product across all steps at once.

The algebra is left unbound here and supplied per instance via `examples.instantiate("examples.mechanics.tricycle.core", PGA2D)` or `PGA3D`, so the same module drives the tricycle in two and three dimensions. The construction follows the tricycle example of ganja.js by Steven De Keninck (https://github.com/enkimute/ganja.js/blob/master/examples/example_chapter11_tricycle.html).
"""

from __future__ import annotations

import numpy as np

from numga import Algebra, concatenate
from numga.backend.context import Context

# Supplied by examples.instantiate.
ga: Algebra
ctx: Context
mv = ctx.multivector
Plane = ga.gatype.vector()                          # Plane+
Point = ga.gatype.antivector()                      # Point-
# Intersection of two hyperplanes: a point in the plane, or a vertical line in space.
Meet = ga.gatype.bivector()                         # Meet+
Motor = ga.gatype.rotor()                           # Motor+


# --- math -----------------------------------------------------------------------------
def axles(wheelbase: float, steering: np.ndarray) -> tuple[Plane, Plane, Motor]:
    """Steering angles `steering` (`[...]`) turn the front axle `wheelbase` ahead of the rear axle about the frame's steering axis, yielding the rear axle (`[] Plane+`), the steered front axle (`[...] Plane+`), and the steering motor (`[...] Motor+`)."""
    rear = mv.x                                                                 # [] Plane+
    rest = mv.x - wheelbase * mv.w                                              # [] Plane+
    axis = rest ^ mv.y                                                          # [] Meet+
    steer = (axis.normalized() * (-steering / 2)).exp()                         # [...] Motor+
    return rear, steer >> rest, steer


def drive(rear: Plane, front: Plane, step: float) -> Motor:
    """Exponentiating each step's turning centre `rear ^ front` (`[n_steps] Meet+`) in the body frame and accumulating on the right gives the tricycle's world-frame poses (`[n_steps] Motor+`)."""
    steps = ((rear ^ front) * (-step / 2)).exp()                                # [n_steps] Motor+
    # Body-frame steps compose on the right; reversing the cumulative left product of reverses gives the cumulative right product.
    return steps.reverse().cumprod(axis=0).reverse()                            # [n_steps] Motor+


def wheels(wheelbase: float, track: float, steer: Motor) -> tuple[Point, Point]:
    """Track width `track`, `wheelbase`, and steering motor `steer` (`[...] Motor+`) place the three ground contacts (`[3] Point-`) and a point slightly ahead along each wheel's rolling heading (`[..., 3] Point-`) in the body frame."""
    ahead = 0.1 * wheelbase
    contacts = ground_point([[0, -track / 2], [0, track / 2], [wheelbase, 0]])  # [3] Point-
    rear_ahead = ground_point([[ahead, -track / 2], [ahead, track / 2]])        # [2] Point-
    front_ahead = steer >> ground_point([wheelbase + ahead, 0])                 # [...] Point-
    forward = concatenate([rear_ahead.broadcast_to(steer.shape + (2,)), front_ahead[..., None]], axis=-1)
    return contacts, forward                                                    # [3] Point-, [..., 3] Point-


# --- plumbing -------------------------------------------------------------------------
def ground_point(coordinates) -> Point:
    """Ground coordinates `coordinates` (`[..., 2]`) along `x` and `y` dualize to unit-weight ground points (`[...] Point-`)."""
    xy = np.asarray(coordinates)
    return (mv.x * xy[..., 0] + mv.y * xy[..., 1] + mv.w).dual()               # [...] Point-
