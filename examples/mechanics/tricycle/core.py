"""A tricycle steered by its front wheel, in the plane and in space, by the same code.

Each axle is a hyperplane of the algebra: a line in the plane, a vertical plane in space. The rear axle
lies along y at the origin; the front axle, at rest, a wheelbase further along x. Steering turns the front
axle about the steering axis, the meet of the front axle at rest with the hyperplane y = 0 down the middle
of the frame: a point in the plane, a vertical line in space.

The two axles meet in the turning centre, `rear ^ front`: a point in the plane, a vertical line in
space. Every wheel rolls around it, since every axle passes through it, and the tricycle moves by its
exponential: a rotation about it, or, with the front wheel straight and the axles parallel, a translation,
the meet then lying at infinity. The meet's weight is the sine of the steering angle and its distance from
the front wheel the wheelbase over that sine, so the front wheel, which the pedals turn, moves at the same
speed at every steering angle.

Over a drive the steering changes from step to step. Each step moves the tricycle in its own frame, by the
exponential of the turning centre at that step's steering, and the steps multiply on the right of the
pose; their running product is the reverse of the running product of their reverses, which multiplies
from the left.

The algebra is not fixed here. `ga` is supplied per instance, by
`examples.instantiate("examples.mechanics.tricycle.core", PGA2D)` or PGA3D, and the same lines drive the
tricycle in the plane and in space.

The construction follows the tricycle example of ganja.js by Steven De Keninck,
https://github.com/enkimute/ganja.js/blob/master/examples/example_chapter11_tricycle.html.

In the notation of vehicle dynamics this reads as Ackermann steering: the instantaneous centre of rotation
lies on the extension of every axle.
"""

from __future__ import annotations

import numpy as np

from numga import Algebra, NumpyContext, concatenate

# Supplied by examples.instantiate.
ga: Algebra
mv = NumpyContext(ga).multivector
Plane = ga.gatype.vector()
Point = ga.gatype.antivector()
# Where two hyperplanes meet: a point in the plane, a line in space.
Meet = ga.gatype.bivector()
Motor = ga.gatype.rotor()


# --- math -----------------------------------------------------------------------------
def axles(wheelbase: float, steering: np.ndarray) -> tuple[Plane, Plane, Motor]:
    """The rear axle, the front axle turned by each steering angle, and the steering motor that turns
    it."""
    rear = mv.x                                                                 # [] Plane
    rest = mv.x - wheelbase * mv.w                                              # [] Plane
    axis = rest ^ mv.y                                                          # [] Meet
    steer = (axis.normalized() * (-steering / 2)).exp()                         # [...] Motor
    return rear, steer >> rest, steer


def drive(rear: Plane, front: Plane, step: float) -> Motor:
    """The tricycle's pose after each step, each step a motion about the turning centre of that step's
    axles, in the tricycle's own frame."""
    steps = ((rear ^ front) * (-step / 2)).exp()                                # [steps] Motor
    return steps.reverse().cumprod(axis=0).reverse()                            # [steps] Motor


def wheels(wheelbase: float, track: float, steer: Motor) -> tuple[Point, Point]:
    """Where each wheel touches the ground, the rear wheels and then the front wheel, and a point a little
    ahead of each along the way it rolls, the front one turned by the steering, in the tricycle's own
    frame."""
    ahead = 0.1 * wheelbase
    contacts = point([[0.0, -track / 2], [0.0, track / 2], [wheelbase, 0.0]])  # [wheels] Point
    rear_ahead = point([[ahead, -track / 2], [ahead, track / 2]])             # [2] Point
    front_ahead = steer >> point([wheelbase + ahead, 0.0])                     # [...] Point
    forward = concatenate([rear_ahead.broadcast_to(steer.shape + (2,)), front_ahead[..., None]], axis=-1)
    return contacts, forward                                                    # [wheels], [..., wheels] Point


# --- plumbing -------------------------------------------------------------------------
def point(coordinates) -> Point:
    """Points on the ground at the given coordinates along x and y."""
    xy = np.asarray(coordinates)
    return (mv.x * xy[..., 0] + mv.y * xy[..., 1] + mv.w).dual()
