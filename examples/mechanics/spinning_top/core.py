"""A spinning top of three ellipsoids on a concave ground, in PGA3D.

Every shape is a quadric, a map from points to planes. The top's parts are solid ellipsoids in its
own frame; the ground is the solid below a shallow paraboloid, a concave quadric. A part's lowest
point against the ground is the pole of its tangent plane facing the ground; how far that point
lies below the ground is its depth, zero for a part in the air. Contact is position based: after a
free step each part is pressed out along the ground's normal by its depth, then held against
sliding up to the friction cone, each a single constraint along one line. The parts are corrected
in sequence so each contact sees the motion caused by the preceding contacts.
The rate is read back from the motor's step.

The contact point is found in one pass: the ground's normal is taken at the part's centre, and
the part's lowest point against that normal is the contact. This is accurate when the part is
more sharply curved than the ground, a small body on a large one, as here. For two quadrics of
comparable curvature the exact contact lies on the pencil of the two, at the point
`(ground - m * part).inverse()(w)` for the parameter m where that point reaches the part's surface.
"""

import jax
import numpy as np

from numga import NumpyContext
from numga.algebras import PGA3D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                               # Scalar+
Point = ga.gatype.antivector()                            # Antivector-
Plane = ga.gatype.vector()                                # Vector+
# A rate of motion, and a force line, the load along it.
Rate = ga.gatype.bivector()                               # Bivector+
Forque = ga.gatype.antibivector()                         # Antibivector-
Inertia = ga.gatype((Forque, Rate))                       # Forque- <- Rate+
InverseInertia = ga.gatype((Rate, Forque))                # Rate+ <- Forque-
Motor = ga.gatype.rotor()                                 # Rotor+
Direction = Point.derive.degenerate                       # Antivector-
Quadric = ga.gatype((Plane, Point))
w = mv.w
axes = mv(Plane.derive.nondegenerate, np.eye(3))


def clamp(value: Scalar, limit: Scalar) -> Scalar:
    """The value with its magnitude held to the limit."""
    magnitude = value.abs()
    # The smaller of the magnitude and the limit.
    least = (magnitude + limit - (magnitude - limit).abs()) * 0.5
    return value * least / (magnitude + 1e-12)


# --- shapes and mass ---------------------------------------------------------------------------
def ellipsoid(centre: Point, semi: np.ndarray) -> Quadric:
    """Solid ellipsoids about centres with the given semi-axes: dyads of the planes through each centre."""
    through = axes - w * (axes & centre[..., None])                           # [..., 3] Plane+
    return (through * (through & Point) / semi**2).sum(axis=-1) - w * (w & Point)


def bowl(curvature: float) -> Quadric:
    """The ground: solid below the paraboloid whose height is the curvature times the squared
    distance from the z axis; negative inside. A bowl for positive curvature, a dome for negative."""
    return 0.5 * (mv.z * (w & Point) + w * (mv.z & Point)) - curvature * (axes[:2] * (axes[:2] & Point)).sum()


def sigma_points(centre: np.ndarray, semi: np.ndarray, mass: np.ndarray) -> tuple[Point, np.ndarray]:
    """Six mass points per solid ellipsoid with its mass, centroid and second moments."""
    offsets = np.eye(3)[None] * semi[:, None, :] * np.sqrt(3 / 5)             # [parts, 3, 3]
    coords = np.concatenate([centre[:, None] + offsets, centre[:, None] - offsets], axis=1)   # [parts, sigma_points, 3]
    return mv.point(coords.reshape(-1, 3)), np.repeat(mass / 6, 6)


def inertia(points: Point, masses: np.ndarray) -> Inertia:
    """The inertia, a map from rates to forques: each mass point joined with its own motion under the
    rate, its momentum, summed over the points."""
    return points.regressive(points.commutator(Rate) * masses).sum()


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
    # Planes to their poles.
    dual = surface.inverse()
    centre = dual(w) / (w & dual(w))
    # The plane through the centre, normal to the direction, and its pole, a direction.
    plane = normal.dual() - w * (normal.dual() & centre)
    conjugate = dual(plane)
    return centre + conjugate / (-(plane & conjugate) * (w & dual(w))).square_root()


def along_ground(direction: Direction, normal: Direction) -> Direction:
    """The direction less its part along the ground's normal: its part in the ground's plane."""
    return direction - normal * (direction.dual() | normal.dual())


def compliance(motor: Motor, inverse_inertia: InverseInertia, lines: Forque) -> tuple[Rate, Scalar]:
    """The body's twist per unit forque along each line, and its compliance along that line."""
    body = motor << lines
    step = inverse_inertia(body)
    return step, step & body


def move(motor: Motor, step: Rate, forque: Scalar) -> Motor:
    """The body carried by the summed twists of the forques."""
    return (motor * ((step * forque).sum(axis=0) * -0.5).exp()).normalized()


def project_contacts(before: Motor, motor: Motor, inverse_inertia: InverseInertia, parts: Quadric, ground: Quadric,
                     static: float, dynamic: float, dt: float) -> tuple[Motor, Rate]:
    """The predicted pose corrected one part after another, so that each sees the corrections before it:
    pressed out by its depth, then held against sliding within the friction cone; then the rate slowed by
    each part's dynamic friction in turn. The parts are taken in a compiled loop, in the motor's
    context."""
    parts = parts.materialize(motor.context)                                # [parts] Plane+ <- Point-

    def touching(motor: Motor, part: Quadric) -> tuple[Point, Direction, Scalar]:
        # One part, as a set of one along a leading axis ahead of any launches, so that the contact helpers sum over
        # the parts and never across launches.
        placed_part = motor[None] >> part[None](motor[None] << Point)             # [1, ...] Plane+ <- Point-
        centre = placed_part.solve(w)                                             # [1] Point-
        contact = lowest_point(placed_part, ground_normal(ground, centre / (w & centre)))
        # The ground's form at the contact over its gradient's length: how far the point lies below the
        # ground, to first order in the depth; a part in the air has none.
        depth = (-(ground(contact) & contact) / (2 * ground(contact).norm())).clip(0, np.inf)
        return contact, ground_normal(ground, contact), depth

    def press(motor: Motor, part: Quadric) -> tuple[Motor, Scalar]:
        contact, normal, depth = touching(motor, part)
        # Press out along the normal line through the contact, by its depth.
        step, line_compliance = compliance(motor, inverse_inertia, contact & normal)
        press_impulse = depth / line_compliance
        motor = move(motor, step, press_impulse)
        # Sticking is possible only when its full correction fits the static friction limit.
        slip = along_ground(contact - (before >> (motor << contact)), normal)
        slip_length = slip.dual().norm()
        step, line_compliance = compliance(motor, inverse_inertia, contact & (-slip / (slip_length + 1e-12)))
        stick_impulse = slip_length / (line_compliance + 1e-12)
        return move(motor, step, stick_impulse * (stick_impulse <= press_impulse * static)), press_impulse

    motor, pressures = jax.lax.scan(press, motor, parts)                    # [] Motor+, [parts, 1] Scalar+
    rate = (before.inverse() * motor).normalized().log() * (-2.0 / dt)

    def slow(rate: Rate, part_and_pressure: tuple[Quadric, Scalar]) -> tuple[Rate, None]:
        part, press_impulse = part_and_pressure
        contact, normal, _ = touching(motor, part)
        # The contact's remaining slip, held by at most the dynamic coefficient times its pressing impulse.
        sliding = along_ground(contact.commutator(motor >> rate), normal)
        speed = sliding.dual().norm()
        step, line_compliance = compliance(motor, inverse_inertia, contact & (-sliding / (speed + 1e-12)))
        return rate + (step * clamp(speed / (line_compliance + 1e-12), press_impulse * (dynamic / dt))).sum(axis=0), None

    rate, _ = jax.lax.scan(slow, rate, (parts, pressures))
    return motor, rate
