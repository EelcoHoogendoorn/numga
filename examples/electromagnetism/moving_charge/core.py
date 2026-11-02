"""Moving charges: the field of a soft charge moving at any speed, from the derivatives of its
potential.

The charge is a Plummer-softened point charge: beyond its core it is a point charge, and its density
is smooth and finite at its centre. At rest its potential points along the lab's time slice and falls
off as one over the distance, and the gradients of the potential and of the field at an event are
maps from steps, antivectors like the events themselves, written in closed form. A moving charge is
the charge at rest seen through a boost: its gradients are the rest-frame ones with the boost
sandwiched around them, the step pulled into the rest frame and the result pushed back out.

The derivative of a gradient pairs each step with its vector by incidence alone,
`(Vector ^ gradients(Antivector)).trace(1, 2)`: of the potential it is the field; of the field it
vanishes, and of the excitation the vacuum makes of the field it is the current, an antivector, which
the lab's time slice reads as the charge density, `mv.t & current`. The metric enters through the
vacuum alone: in its map from fields to excitations, and in the distances of the charge's own
potential. Each derivative takes the half of the geometric derivative that carries anything: the full
one, `(Vector * gradients(Antivector)).trace(1, 2)`, would add the potential's divergence, zero because
the charges' potentials are in the Lorenz gauge, and the field's own derivative, zero because the field
comes from a potential.

A point charge circling at constant speed radiates. Its potential at an event is set by where the
charge was when it sent the light that reaches the event, the retarded time, and the gradient of
that potential is again a closed-form map: its derivative is the full field, the bound field that
moves with the charge and the waves it sheds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from numga import NumpyContext
from numga.algebras import STA as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Vector = ga.gatype.vector()
Bivector = ga.gatype.bivector()
Antivector = ga.gatype.antivector()
Rotor = ga.gatype.rotor()
PotentialGradient = ga.gatype((Vector, Antivector))          # Vector+ <- Antivector-
FieldGradient = ga.gatype((Bivector, Antivector))            # Bivector+ <- Antivector-
# The retarded time: steps that always close in, by at least the orbit's speed each, then Newton's.
n_settle_steps = 6
n_newton_steps = 6

# The vacuum: from a field to the excitation it causes.
vacuum = Bivector.dual()                                     # Antibivector- <- Bivector+
# One unit along the lab's time: the step across the lab's slice of space `mv.t`.
time = -mv.t.dual()                                          # [] Antivector-


@dataclass(frozen=True)
class Orbit:
    charge: float
    radius: float
    # As a fraction of the speed of light.
    speed: float


@dataclass(frozen=True)
class Charge:
    # The strength of each charge.
    charge: np.ndarray                                       # [...]
    core_radius: float
    # The rest frame seen from the lab.
    boost: Rotor                                             # [...] Rotor+
    # Where the charge is at lab time zero.
    position: Antivector                                     # [...] Antivector-


# --- math -----------------------------------------------------------------------------
def potential_gradient(charge: Charge, events: Antivector) -> PotentialGradient:
    """The change in the potential for a small step from each event."""
    separations, weight, _ = _profile(charge, events)                        # [...] Antivector-, Scalar+
    # At rest the potential points along the time slice and changes with the spatial distance.
    at_rest = weight * mv.t * (separations | Antivector)                      # [...] Vector+ <- Antivector-
    return charge.boost >> at_rest(charge.boost << Antivector)                # [...] Vector+ <- Antivector-


def field_gradient(charge: Charge, events: Antivector) -> FieldGradient:
    """The change in the field for a small step from each event."""
    separations, weight, weight_change = _profile(charge, events)            # [...] Antivector-, Scalar+, Scalar+
    # The radial field changes with the distance and turns with the step.
    # Each is the plane of a step and time, their join, which the dual turns into a field.
    radial = weight_change * (separations & time).dual() * (separations | Antivector)   # [...] Bivector+ <- Antivector-
    turning = weight * (Antivector & time).dual()                            # [...] Bivector+ <- Antivector-
    return charge.boost >> (radial + turning)(charge.boost << Antivector)     # [...] Bivector+ <- Antivector-


def field(gradients: PotentialGradient) -> Bivector:
    """The derivative of the potential: each step's vector wedged with the potential's change along it."""
    return (Vector ^ gradients(Antivector)).trace(1, 2)                      # [...] Bivector+


def current(gradients: FieldGradient) -> Antivector:
    """The derivative of the excitation the vacuum makes of the field: Maxwell's equation with sources."""
    return (Vector ^ vacuum(gradients(Antivector))).trace(1, 2)              # [...] Antivector-


def orbit_potential_gradient(orbit: Orbit, events: Antivector) -> PotentialGradient:
    """The change in the circling charge's potential for a small step from each event."""
    emitted = retarded(orbit, events)                                        # [...] Scalar+
    position, tangent, bend = worldline(orbit, emitted)                      # [...] Antivector-, Antivector-, Antivector-
    # The charge's velocity and acceleration in its own time, and the light ray from it to the event.
    dilation = 1 / np.sqrt(1 - orbit.speed**2)
    velocity, acceleration = dilation * tangent, dilation**2 * bend          # [...] Antivector-, Antivector-
    ray = events - position                                                  # [...] Antivector-
    # The charge's own slice of space, which reads its velocity as one: minus the velocity's dual, as
    # for the lab's `time`; and how the slice turns with the acceleration.
    rest_slice, slice_change = -velocity.dual(), -acceleration.dual()      # [...] Vector+, Vector+
    # The distance as the charge saw it when it sent the light: the ray's extent, as its own slice reads it.
    rest_distance = rest_slice & ray                                         # [...] Scalar+
    # A step moves the emission along the ray by what the ray's slice, the light cone's tangent, reads of
    # it; that changes the slice and the distance.
    emission = (-ray.dual() & Antivector) / rest_distance                    # [...] Scalar+ <- Antivector-
    distance_change = (rest_slice & Antivector) + ((slice_change & ray) - 1) * emission   # [...] Scalar+ <- Antivector-
    weight = orbit.charge / (4 * np.pi) / rest_distance                      # [...] Scalar+
    # As at rest, the potential is minus the charge's slice, here over the distance.
    return -weight * (slice_change * emission - rest_slice * distance_change / rest_distance)   # [...] Vector+ <- Antivector-


def worldline(orbit: Orbit, times: Scalar) -> tuple[Antivector, Antivector, Antivector]:
    """The circling charge at each lab time, its change per unit lab time, and that change's change."""
    # The angular rate.
    turning = orbit.speed / orbit.radius
    # The offset turned in the orbit's plane, and moving under that turn by its commutator with the plane.
    offset = (mv.xy * (-0.5 * turning) * times).exp() >> (mv.x.dual() * orbit.radius)   # [...] Antivector-
    tangent = time + turning * offset.commutator(mv.xy)                      # [...] Antivector-
    bend = turning * (tangent - time).commutator(mv.xy)                      # [...] Antivector-
    return times * time + offset, tangent, bend


def retarded(orbit: Orbit, events: Antivector) -> Scalar:
    """The lab time at which the charge sent the light that reaches each event: a few steps that
    move the time back by the distance light covers, which always close in, then Newton's steps."""
    def light(times: Scalar) -> tuple[Scalar, Scalar, Scalar]:
        """How far light from the charge at each time falls short of the event, its rate, and the distance."""
        position, tangent, _ = worldline(orbit, times)                       # [...] Antivector-, Antivector-, Antivector-
        ray = events - position                                              # [...] Antivector-
        across = ray - (mv.t & ray) * time                                   # [...] Antivector-
        distance = (across | across).square_root()                           # [...] Scalar+
        return (mv.t & events) - times - distance, 1 - (across | tangent) / distance, distance

    times = mv.t & events                                                    # [...] Scalar+
    for _ in range(n_settle_steps):
        _, _, distance = light(times)
        times = (mv.t & events) - distance
    for _ in range(n_newton_steps):
        shortfall, rate, _ = light(times)
        times = times + shortfall / rate
    return times


# The circling charge's field at an event in numga's shader dialect, for pictures that ask for it at every sample of
# every pixel: the worldline, the retarded time and the derivative of the potential's gradient, as above. The gradient
# is a sum of vectors, each times a form that reads the step, `rest_slice * (distance_change & step)`, and the
# derivative of each such term wedges its form with its vector, `distance_change ^ rest_slice`, which is what the trace
# of `Vector ^ gradients(Antivector)` sums to.
orbit_field_source = f"""
// One unit along the lab's time, the steps of the retarded time's search, and pi.
const time = -mv.t.dual();
const n_settle_steps = {n_settle_steps}u;
const n_newton_steps = {n_newton_steps}u;
const pi = {np.pi!r};
// The orbit: its charge, its radius, and its speed as a fraction of the speed of light.
var<uniform> orbit_charge: f32;
var<uniform> orbit_radius: f32;
var<uniform> orbit_speed: f32;

struct Worldline {{
    position: Antivector,
    tangent: Antivector,
    bend: Antivector,
}}

// The circling charge at a lab time, its change per unit lab time, and that change's change.
fn worldline(lab_time: f32) -> Worldline {{
    // The angular rate.
    let turning = orbit_speed / orbit_radius;
    // The offset turned in the orbit's plane, and moving under that turn by its commutator with the plane.
    let offset = (mv.xy * (-0.5 * turning * lab_time)).exp() >> (mv.x.dual() * orbit_radius);
    let tangent = time + turning * offset.commutator(mv.xy);
    let bend = turning * (tangent - time).commutator(mv.xy);
    return Worldline(lab_time * time + offset, tangent, bend);
}}

struct Light {{
    shortfall: f32,
    rate: f32,
    reach: f32,
}}

// How far light from the charge at a lab time falls short of the event, its rate, and the distance it covers.
fn light(event: Antivector, lab_time: f32) -> Light {{
    let charge = worldline(lab_time);
    let ray = event - charge.position;
    let across = ray - (mv.t & ray) * time;
    let reach = (across | across).square_root();
    return Light((mv.t & event) - lab_time - reach, 1 - (across | charge.tangent) / reach, reach);
}}

// The lab time at which the charge sent the light that reaches the event: a few steps that move the time back by the
// distance light covers, which always close in, then Newton's steps.
fn retarded(event: Antivector) -> f32 {{
    var emitted = mv.t & event;
    for (var index = 0u; index < n_settle_steps; index++) {{
        emitted = (mv.t & event) - light(event, emitted).reach;
    }}
    for (var index = 0u; index < n_newton_steps; index++) {{
        let seen = light(event, emitted);
        emitted = emitted + seen.shortfall / seen.rate;
    }}
    return emitted;
}}

// The field at an event: the derivative of the circling charge's potential.
fn orbit_field(event: Antivector) -> Bivector {{
    let charge = worldline(retarded(event));
    // The charge's velocity and acceleration in its own time, and the light ray from it to the event.
    let dilation = 1 / sqrt(1 - orbit_speed * orbit_speed);
    let velocity = dilation * charge.tangent;
    let acceleration = dilation * dilation * charge.bend;
    let ray = event - charge.position;
    // The charge's own slice of space, which reads its velocity as one, and how it turns with the acceleration.
    let rest_slice = -velocity.dual();
    let slice_change = -acceleration.dual();
    // The distance as the charge saw it when it sent the light.
    let rest_distance = rest_slice & ray;
    // What a step moves the emission by, and the distance: forms, the slices that read the step.
    let emission = -ray.dual() / rest_distance;
    let distance_change = rest_slice + ((slice_change & ray) - 1) * emission;
    let weight = orbit_charge / (4 * pi) / rest_distance;
    // The potential's gradient takes a step to
    // `-weight * (slice_change * (emission & step) - rest_slice * (distance_change & step) / rest_distance)`.
    return -weight * ((emission ^ slice_change) - (distance_change ^ rest_slice) / rest_distance);
}}
"""


def _profile(charge: Charge, events: Antivector) -> tuple[Antivector, Scalar, Scalar]:
    """The spatial separation from the charge in its rest frame, the field per unit separation, and
    that weight's change per unit of `separation | step`."""
    at_rest = charge.boost << (events - charge.position)                     # [...] Antivector-
    separations = at_rest - (mv.t & at_rest) * time                          # [...] Antivector-
    softened = charge.core_radius**2 + (separations | separations)           # [...] Scalar+
    weight = charge.charge / (4 * np.pi) / (softened * softened.square_root())   # [...] Scalar+
    return separations, weight, -3 * weight / softened


# --- plumbing -------------------------------------------------------------------------
def grid(half_width: float, half_height: float, columns: int, rows: int) -> Antivector:
    """Events at lab time zero on a grid in the plane z = 0, at the centres of its cells."""
    x = (np.arange(columns) + 0.5) / columns * 2 * half_width - half_width
    y = (np.arange(rows) + 0.5) / rows * 2 * half_height - half_height
    return (mv.x.dual() * x[None, :] + mv.y.dual() * y[:, None]).cast(Antivector)   # [rows, columns] Antivector-
