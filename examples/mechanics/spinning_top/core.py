"""A spinning top of three ellipsoids on a concave ground, in PGA3D.

Every shape is a quadric, a map from points to planes. The top's parts are solid ellipsoids in its
own frame; the ground is the solid below a shallow paraboloid, a concave quadric. A part's lowest
point against the ground is the pole of its tangent plane facing the ground; how far that point
lies below the ground is its depth, zero for a part in the air. Contact is position based: after a
free step each part is pressed out along the ground's normal by its depth, then held against
sliding up to the friction cone and against turning about the normal up to the drilling limit,
each a single constraint along one line, batched over the parts.
The rate is read back from the motor's step.

The contact point is found in one pass: the ground's normal is taken at the part's centre, and
the part's lowest point against that normal is the contact. This is accurate when the part is
more sharply curved than the ground, a small body on a large one, as here. For two quadrics of
comparable curvature the exact contact lies on the pencil of the two, p = (ground - m part)^-1 w,
at the parameter m where p reaches the part's surface.
"""

import numpy as np

from numga import NumpyContext
from numga.algebras import PGA3D as ga
from examples.geometry.surface_curvature.core import principal

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Point = ga.gatype.antivector()
Plane = ga.gatype.vector()
Line = ga.gatype.bivector()
Motor = ga.gatype.rotor()
Direction = ga.gatype(ga.subspace.antivector().degenerate())
Quadric = ga.gatype((Plane, Point))
w = mv.w
axes = mv("x y z", np.eye(3))


def point(coords: np.ndarray) -> Point:
    return (mv("x y z", coords) + w).dual()


def clamp(value: Scalar, limit: Scalar) -> Scalar:
    """The value with its magnitude held to the limit."""
    magnitude = value.abs()
    least = (magnitude + limit - (magnitude - limit).abs()) * 0.5          # min(|value|, limit)
    return value * least / (magnitude + 1e-12)


# --- shapes and mass ---------------------------------------------------------------------------
def ellipsoid(centre: Point, semi: np.ndarray) -> Quadric:
    """Solid ellipsoids about centres with the given semi-axes: dyads of the planes through each centre."""
    through = axes - w * (axes & centre[..., None])                           # [..., 3] Plane
    return (through * (through & Point) / semi**2).sum(axis=-1) - w * (w & Point)


def bowl(curvature: float) -> Quadric:
    """The ground: solid below z = curvature (x^2 + y^2); negative inside. A bowl for positive
    curvature, a dome for negative."""
    return 0.5 * (mv.z * (w & Point) + w * (mv.z & Point)) - curvature * (axes[:2] * (axes[:2] & Point)).sum()


def sigma_points(centre: np.ndarray, semi: np.ndarray, mass: np.ndarray):
    """Six mass points per solid ellipsoid with its mass, centroid and second moments."""
    offsets = np.eye(3)[None] * semi[:, None, :] * np.sqrt(3 / 5)             # [parts, 3, 3]
    coords = np.concatenate([centre[:, None] + offsets, centre[:, None] - offsets], axis=1)   # [parts, sigma_points, 3]
    return point(coords.reshape(-1, 3)), np.repeat(mass / 6, 6)


def inertia(points: Point, masses: np.ndarray):
    """Momentum <- rate: each mass point joined with its own motion under an open rate."""
    return (points & points.commutator(Line) * masses).sum()


# --- contact -----------------------------------------------------------------------------------
def ground_normal(ground: Quadric, points: Point) -> Direction:
    """The ground's unit normal at points, toward the air, where its form grows.

    A point's polar plane under the ground quadric is the gradient of the ground's form there;
    read as a direction and divided by its length it is the normal, at any point, on the ground
    or off it.
    """
    return ground(points).dual().cast(Direction) / ground(points).norm()


def lowest_point(surface: Quadric, normal: Direction) -> Point:
    """Each quadric's point furthest against the normal: the pole of its tangent plane on that side.

    The plane through the quadric's centre normal to the given direction has its pole at infinity:
    the direction conjugate to it, along which the centre reaches the planes tangent to the
    quadric parallel to it. Scaled to reach the surface, it leads from the centre to the lowest
    point. For a part against the ground, the normal is the ground's at the part's centre, so the
    result is the true deepest point only while the ground turns little across the part.
    """
    dual = surface.inverse()                                                # planes to their poles
    centre = dual(w) / (w & dual(w))
    plane = normal.dual() - w * (normal.dual() & centre)                    # through the centre, normal to it
    conjugate = dual(plane)                                                 # its pole, a direction
    return centre + conjugate / (-(plane & conjugate) * (w & dual(w))).square_root()


def along_ground(direction: Direction, normal: Direction) -> Direction:
    """The direction less its part along the ground's normal: its part in the ground's plane."""
    return direction - normal * (direction.dual() | normal.dual())


def compliance(motor: Motor, I_inv, lines: Line) -> tuple[Line, Scalar]:
    """The body's twist per unit wrench along each line, and its compliance along that line."""
    body = motor << lines
    step = I_inv(body)
    return step, step & body


def move(motor: Motor, step: Line, wrench: Scalar) -> Motor:
    """Carry the body by the summed twists of the wrenches."""
    return (motor * ((step * wrench).sum(axis=0) * -0.5).exp()).normalized()


def project_contacts(before: Motor, motor: Motor, I_inv, parts: Quadric, ground: Quadric,
                     static: float, dynamic: float, indentation: float, dt: float) -> tuple[Motor, Line]:
    """Correct the predicted pose: press each part out by its depth, then hold it against sliding
    and against turning about the normal, within the friction cone and the drilling limit."""
    placed = motor >> parts(motor << Point)                                 # [parts] in the world
    centre = placed.solve(w)                                                # the pole of the plane at infinity
    # Each part's lowest point against the ground's normal at its centre is its contact point.
    contact = lowest_point(placed, ground_normal(ground, centre / (w & centre)))
    normal = ground_normal(ground, contact)
    # The ground's form at the contact over its gradient's length: how far the point lies below the
    # ground, to first order in the depth; a part in the air has none.
    depth = (-(ground(contact) & contact) / (2 * ground(contact).norm())).clip(0, np.inf)

    # Press out along the normal line through each contact, by its depth.
    step, give = compliance(motor, I_inv, contact & normal)
    pressed = depth / give
    motor = move(motor, step, pressed)

    # Hold against sliding: back toward where the contact's material point was, in the ground's plane.
    slid = along_ground((contact - (before >> (motor << contact))).cast(Direction), normal)
    back = slid.dual().norm()
    step, give = compliance(motor, I_inv, contact & (-slid / (back + 1e-12)))
    motor = move(motor, step, clamp(back / give, pressed * static))

    # Hold against turning about the normal: undo the step's turn, with at most static friction times the
    # pressing wrench times the contact patch's radius, sqrt(radius of curvature * indentation).
    couple = normal.dual() ^ w                                              # the torque about the normal
    turned = couple & ((motor * ~before).log() * -2.0)
    curvatures = principal(placed, contact)
    patch = ((2 / (curvatures[..., 0] + curvatures[..., 1])).abs() * indentation).square_root()
    step, give = compliance(motor, I_inv, couple)
    motor = move(motor, step, clamp(-turned / give, pressed * static * patch))

    # Dynamic friction: the rate the corrected step implies, less the contacts' sliding velocity, with
    # at most the dynamic coefficient times the normal impulse, the pressing wrench over the step.
    rate = (~before * motor).log() * (-2.0 / dt)
    sliding = along_ground(contact.commutator(motor >> rate).cast(Direction), normal)
    speed = sliding.dual().norm()
    step, give = compliance(motor, I_inv, contact & (-sliding / (speed + 1e-12)))
    return motor, rate + (step * clamp(speed / give, pressed * dynamic / dt)).sum(axis=0)
