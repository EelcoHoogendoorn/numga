"""A spinning top of solid ellipsoids precesses on a quadric ground in 3D projective geometric algebra.

Every shape is modelled as a quadric, a duality map from points to planes whose pairing with a point is
negative inside the solid: the top's three parts are solid ellipsoids in the body frame, and the ground
is the solid below a paraboloid. A part's lowest point against the ground is the pole of its tangent
plane facing the ground, and how far that point lies below the ground is its penetration depth. Contact
is position based: after an unconstrained Lie-group step, each part is pressed out along the ground
normal by its depth and held against sliding within the static friction cone, each correction acting
along a single force line weighted by that line's rotational and translational compliance. Dynamic
friction then slows the remaining tangential slip on the implied body rate, part by part.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import jax
import jax.numpy as jnp
import numpy as np

from numga import NumpyContext, concatenate, stack
from numga.algebras import PGA3D as ga
from numga.backend.jax import JaxContext
from examples.mechanics import lie_integrators as lie

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                    # Scalar+
Point = ga.gatype.antivector()                                                 # Antivector-
Plane = ga.gatype.vector()                                                     # Vector+
Direction = Point.derive.degenerate                                            # Antivector-
Rate = ga.gatype.bivector()                                                    # Bivector+
Forque = ga.gatype.antibivector()                                              # Antibivector-
Inertia = ga.gatype((Forque, Rate))                                            # Forque- <- Rate+
InverseInertia = ga.gatype((Rate, Forque))                                     # Rate+ <- Forque-
Motor = ga.gatype.rotor()                                                      # Rotor+
Quadric = ga.gatype((Plane, Point))                                            # Plane+ <- Point-
w = mv.w                                                                       # Plane+
axes = mv(Plane.derive.nondegenerate, np.eye(3))                               # [3] Plane+
directions = axes.dual()                                                       # [3] Antivector-
origin = w.dual()                                                              # [] Point-
up = directions[2]                                                             # [] Antivector-


# --- math -----------------------------------------------------------------------------
def ellipsoid(centres: Point, semi_axes: np.ndarray) -> Quadric:
    """Solid ellipsoids centred on `centres` with semi-axes `semi_axes` (`[..., 3]`) form quadric maps (`[...] Plane+ <- Point-`)."""
    planes = axes - w * (axes & centres[..., None])                            # [..., 3] Plane+
    return (planes * (planes & Point) / semi_axes**2).sum(axis=-1) - w * (w & Point)


def bowl(curvature: float) -> Quadric:
    """The ground below a paraboloid of height `curvature` times the squared radial distance forms a quadric (`[] Plane+ <- Point-`)."""
    return 0.5 * (axes[2] * (w & Point) + w * (axes[2] & Point)) - curvature * (axes[:2] * (axes[:2] & Point)).sum()


def top_body(part_heights: np.ndarray, semi_axes: np.ndarray, part_masses: np.ndarray) -> tuple[Quadric, Inertia, np.ndarray]:
    """Six mass points per solid ellipsoid place the parts about their centre of mass (`[..., 3] Plane+ <- Point-`) and sum their momenta into the inertia (`[...] Forque- <- Rate+`)."""
    centre_height = (part_heights * part_masses).sum(axis=-1) / part_masses.sum(axis=-1)
    centres = origin + up * (part_heights - centre_height[..., None])          # [..., 3] Point-
    # Six mass points per part along its principal axes match the solid ellipsoid's mass, centroid, and second moments.
    offsets = directions * (semi_axes * np.sqrt(3 / 5))                        # [..., 3, 3] Antivector-
    mass_points = concatenate([centres[..., None] + offsets, centres[..., None] - offsets], axis=-1)  # [..., 3, 6] Point-
    # Joining each mass point with its velocity under a rate gives its momentum map; summing yields the body inertia.
    momenta = (mass_points * (part_masses[..., None] / 6)).regressive(mass_points.commutator(Rate))   # [..., 3, 6] Forque- <- Rate+
    return ellipsoid(centres, semi_axes), momenta.sum(axis=-1).sum(axis=-1), centre_height


def ground_normal(ground: Quadric, points: Point) -> Direction:
    """Each point's polar plane under `ground` gives the outward unit normal direction toward the air (`[...] Antivector-`)."""
    return ground(points).dual().cast(Direction) / ground(points).norm()


def lowest_point(surface: Quadric, normal: Direction) -> Point:
    """The pole of each quadric's tangent plane facing against `normal` locates its lowest point (`[...] Point-`)."""
    # Inverting the quadric maps planes to their poles; the pole of the plane at infinity is the quadric's centre.
    dual_quadric = surface.inverse()                                           # [...] Point- <- Plane+
    centre = dual_quadric(w) / (w & dual_quadric(w))                           # [...] Point-
    # The polar of the central plane perpendicular to `normal` is the conjugate direction toward the tangent point.
    central_plane = normal.dual() - w * (normal.dual() & centre)               # [...] Plane+
    conjugate = dual_quadric(central_plane)                                    # [...] Antivector-
    return centre + conjugate / (-(central_plane & conjugate) * (w & dual_quadric(w))).square_root()


def contact_state(motor: Motor, parts: Quadric, ground: Quadric) -> tuple[Point, Direction, Scalar]:
    """Each placed part's lowest point against the ground normal at its centre yields the contact point, normal, and signed penetration depth."""
    placed_parts = motor[..., None] >> parts(motor[..., None] << Point)        # [..., ...] Plane+ <- Point-
    centres = placed_parts.solve(w)                                            # [..., ...] Point-
    contact = lowest_point(placed_parts, ground_normal(ground, centres / (w & centres)))  # [..., ...] Point-
    # Dividing the ground's quadratic form by twice its gradient length measures depth to first order.
    depth = -(ground(contact) & contact) / (2 * ground(contact).norm())        # [..., ...] Scalar+
    return contact, ground_normal(ground, contact), depth


def ground_tangent(direction: Direction, normal: Direction) -> Direction:
    """Subtracting the component along `normal` projects `direction` onto the ground's tangent plane (`[...] Antivector-`)."""
    return direction - normal * (direction.dual() | normal.dual())


def clamp(value: Scalar, limit: Scalar) -> Scalar:
    """Clamping holds the magnitude of `value` at most equal to `limit` (`[...] Scalar+`)."""
    magnitude = value.abs()                                                    # [...] Scalar+
    smaller = (magnitude + limit - (magnitude - limit).abs()) / 2              # [...] Scalar+
    return value * smaller / (magnitude + 1e-12)


def line_compliance(motor: Motor, inverse_inertia: InverseInertia, lines: Forque) -> tuple[Rate, Scalar]:
    """Pulling `lines` into the body frame maps unit loads to body twists (`[...] Rate+`) and scalar compliances (`[...] Scalar+`)."""
    body_lines = motor << lines                                                # [...] Forque-
    twists = inverse_inertia(body_lines)                                       # [...] Rate+
    return twists, twists & body_lines


def impulse_step(motor: Motor, twists: Rate, impulses: Scalar) -> Motor:
    """Applying `impulses` along `twists` steps `motor` by the summed rigid-body displacement (`[...] Rotor+`)."""
    step_motor = ((twists * impulses).sum(axis=-1) * -0.5).exp()               # [...] Rotor+
    return (motor * step_motor).normalized()


def project_contacts(before: Motor, motor: Motor, inverse_inertia: InverseInertia, parts: Quadric, ground: Quadric,
                     static: float, dynamic: np.ndarray, dt: float) -> tuple[Motor, Rate]:
    """Sequential Gauss-Seidel impulses press each penetrating part out of `ground` and damp tangential slip within the friction cone."""
    part_inertia = inverse_inertia[..., None]                                  # [..., 1] Rate+ <- Forque-
    # Materializing in the motor's context lets `jax.lax.scan` step through the parts in sequence.
    parts = parts.materialize(motor.context)                                   # [..., ...] Plane+ <- Point-
    dynamic_limit = jnp.asarray(dynamic)[..., None] / dt

    def press_step(motor: Motor, part: Quadric) -> tuple[Motor, Scalar]:
        # Trailing singleton axes handle one contact part at a time across any leading batch of bodies.
        contact, normal, depth = contact_state(motor, part[..., None], ground)
        # Normal projection presses the contact out along the ground normal by its positive depth.
        twists, compliance = line_compliance(motor[..., None], part_inertia, contact & normal)
        press_impulse = depth.clip(0, np.inf) / compliance                     # [..., 1] Scalar+
        motor = impulse_step(motor, twists, press_impulse)                     # [...] Rotor+
        # Static friction restores the contact's tangential position when the impulse fits inside the static cone.
        previous_contact = before[..., None] >> (motor[..., None] << contact)  # [..., 1] Point-
        slip = ground_tangent(contact - previous_contact, normal)              # [..., 1] Antivector-
        slip_length = slip.dual().norm()                                       # [..., 1] Scalar+
        twists, compliance = line_compliance(motor[..., None], part_inertia, contact & (-slip / (slip_length + 1e-12)))
        stick_impulse = slip_length / (compliance + 1e-12)                     # [..., 1] Scalar+
        return impulse_step(motor, twists, stick_impulse * (stick_impulse <= press_impulse * static)), press_impulse

    motor, press_impulses = jax.lax.scan(press_step, motor, parts)
    # The relative motor from `before` to the projected `motor` determines the post-projection body rate.
    rate = (before.inverse() * motor).normalized().log() * (-2 / dt)           # [...] Rate+

    def friction_step(rate: Rate, part_and_impulse: tuple[Quadric, Scalar]) -> tuple[Rate, None]:
        part, press_impulse = part_and_impulse
        contact, normal, _ = contact_state(motor, part[..., None], ground)
        # Dynamic friction opposes the remaining tangential velocity up to `dynamic` times the normal impulse.
        sliding = ground_tangent(contact.commutator(motor[..., None] >> rate[..., None]), normal)
        speed = sliding.dual().norm()                                          # [..., 1] Scalar+
        twists, compliance = line_compliance(motor[..., None], part_inertia, contact & (-sliding / (speed + 1e-12)))
        drag_impulse = clamp(speed / (compliance + 1e-12), press_impulse * dynamic_limit)
        return rate + (twists * drag_impulse).sum(axis=-1), None

    rate, _ = jax.lax.scan(friction_step, rate, (parts, press_impulses))
    return motor, rate


def spin(motor: Motor, rate: Rate, parts: Quadric, inertia: Inertia, masses: np.ndarray, ground: Quadric,
         gravity: float, drag: float, static_friction: float, dynamic_friction: np.ndarray,
         dt: float, n_frames: int, n_steps_per_frame: int) -> Iterator[tuple[Motor, Rate]]:
    """Yields the body poses and rates (`[...] Rotor+`, `[...] Rate+`) across `n_frames` frames spaced `n_steps_per_frame` steps apart."""
    inverse_inertia = inertia.inverse()                                        # [...] Rate+ <- Forque-
    # Scanning in `jax.lax.scan` moves the part axis to axis 0 so each scan iteration sees one part across all tops.
    scan_parts = stack([parts[..., index] for index in range(parts.shape[-1])])

    def external_forque(motor: Motor, rate: Rate) -> Forque:
        # Gravity acts downward through the centre of mass while air drag opposes the body's angular and linear momentum.
        weight = motor << ((motor >> origin) & (up * (-gravity * masses)))     # [...] Forque-
        return weight - inertia(rate) * drag

    def substep(state: tuple[Motor, Rate], _: None) -> tuple[tuple[Motor, Rate], None]:
        before, rate = state
        predicted, rate = lie.explicit_rk4(before, rate, inertia, inverse_inertia, dt, external_forque)
        projected, rate = project_contacts(before, predicted.normalized(), inverse_inertia, scan_parts, ground,
                                           static_friction, dynamic_friction, dt)
        return (projected.retyped(Motor), rate.retyped(Rate)), None

    frame_step: Callable = jax.jit(lambda state: jax.lax.scan(substep, state, length=n_steps_per_frame)[0])
    context = JaxContext(ga)
    state = (motor.retyped(Motor).materialize(context), rate.retyped(Rate).materialize(context))
    for _ in range(n_frames):
        yield state[0].materialize(mv.context), state[1].materialize(mv.context)
        state = frame_step(state)


def lean_angle(motor: Motor) -> Scalar:
    """The angle between the body's symmetry axis and the vertical `up` measures the top's lean (`[...] Scalar+`)."""
    return ((motor >> up).dual() | up.dual()).abs().arccos()
