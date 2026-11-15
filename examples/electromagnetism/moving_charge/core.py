"""A soft charge moving at constant velocity and a point charge circling at constant speed produce
electromagnetic fields determined by the spacetime derivatives of their potentials.

Beyond its softening core a Plummer charge matches a point charge while keeping a smooth, finite
density at its centre. At rest its potential points along the lab's time slice and falls off with
distance, giving closed-form maps from spacetime steps (`Antivector-`) to changes in the potential and
in the field. Boosting that rest-frame map pulls each step into the rest frame and pushes the
resulting change back to the lab, `boost >> rest_gradient(boost << Antivector)`.

Contracting each basis vector with a gradient's response along the dual step,
`(Vector ^ gradients(Antivector)).trace(1, 2)`, takes the exterior derivative by incidence alone.
Applied to the potential it yields the field (`Bivector+`), applied to the field it vanishes, and
applied to the vacuum excitation `vacuum(field)` (`Antibivector-`) it yields the four-current
(`Antivector-`), whose pairing `t & current` with the lab's time slice is the charge density. The
metric enters only through the vacuum duality map and through the rest-frame distances of the source,
while the Lorenz gauge makes the omitted divergence part of `(Vector * gradients(Antivector)).trace(1, 2)`
identically zero.

For a point charge on a circular orbit, each event receives light emitted at an earlier retarded time
found by fixed-point light-cone contraction followed by Newton steps. Differentiating the resulting
retarded potential map produces both the bound field carried with the charge and the radiation waves
shed outward.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext
from numga.algebras import STA as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Vector = ga.gatype.vector()
Bivector = ga.gatype.bivector()
Antivector = ga.gatype.antivector()
Rotor = ga.gatype.rotor()
PotentialGradient = ga.gatype((Vector, Antivector))                          # Vector+ <- Antivector-
FieldGradient = ga.gatype((Bivector, Antivector))                            # Bivector+ <- Antivector-

# Fixed-point light-cone contractions always close in at subluminal speed before Newton steps polish the root.
n_settle_steps = 6
n_newton_steps = 6

# Vacuum constitutive map from fields to excitations, the one material map where the spacetime metric enters.
vacuum = Bivector.dual()                                                     # [] Antibivector- <- Bivector+
# Unit future-pointing lab time step across the lab's spatial slice `mv.t` (`mv.t & time == 1`).
time = -mv.t.dual()                                                          # [] Antivector-


# --- math -----------------------------------------------------------------------------
def charge_profile(
    strengths: np.ndarray,
    core_radius: float,
    boosts: Rotor,
    positions: Antivector,
    events: Antivector,
) -> tuple[Antivector, Scalar, Scalar]:
    """Pulling `events` into each charge's rest frame gives the spatial separation (`[...] Antivector-`),
    the Coulomb weight per unit separation (`[...] Scalar+`), and that weight's radial slope (`[...] Scalar+`)."""
    # Undo the boost to read each event relative to the charge in the charge's own rest frame, then project out time.
    rest_offsets = boosts << (events - positions)                            # [...] Antivector-
    separations = rest_offsets - (mv.t & rest_offsets) * time                # [...] Antivector-
    squared_distance = core_radius**2 + (separations | separations)          # [...] Scalar+
    weight = strengths / (4 * np.pi) / (squared_distance * squared_distance.square_root())  # [...] Scalar+
    return separations, weight, -3 * weight / squared_distance


def potential_gradient(
    strengths: np.ndarray,
    core_radius: float,
    boosts: Rotor,
    positions: Antivector,
    events: Antivector,
) -> PotentialGradient:
    """Boosting the rest-frame Coulomb slope maps a spacetime step at each event to the change in
    four-potential (`[...] Vector+ <- Antivector-`)."""
    separations, weight, _ = charge_profile(strengths, core_radius, boosts, positions, events)  # [...] Antivector-, Scalar+
    # At rest the potential lies along `mv.t` and varies only with spatial distance from the charge.
    rest_gradient = weight * mv.t * (separations | Antivector)               # [...] Vector+ <- Antivector-
    return boosts >> rest_gradient(boosts << Antivector)                     # [...] Vector+ <- Antivector-


def field_gradient(
    strengths: np.ndarray,
    core_radius: float,
    boosts: Rotor,
    positions: Antivector,
    events: Antivector,
) -> FieldGradient:
    """Boosting the rest-frame electric-field slope maps a spacetime step at each event to the change
    in field (`[...] Bivector+ <- Antivector-`)."""
    separations, weight, weight_slope = charge_profile(strengths, core_radius, boosts, positions, events)  # [...] Antivector-, Scalar+, Scalar+
    # At rest the field is purely electric, `(separations & time).dual()`; a step changes its radial weight and turns its direction.
    radial_term = weight_slope * (separations & time).dual() * (separations | Antivector)  # [...] Bivector+ <- Antivector-
    transverse_term = weight * (Antivector & time).dual()                    # [...] Bivector+ <- Antivector-
    return boosts >> (radial_term + transverse_term)(boosts << Antivector)   # [...] Bivector+ <- Antivector-


def field(gradients: PotentialGradient) -> Bivector:
    """Wedge-contracting each basis vector with `gradients` gives the electromagnetic field (`[...] Bivector+`)."""
    return (Vector ^ gradients(Antivector)).trace(1, 2)                      # [...] Bivector+


def four_current(gradients: FieldGradient) -> Antivector:
    """Wedge-contracting each basis vector with the vacuum excitation change gives the source four-current
    (`[...] Antivector-`)."""
    return (Vector ^ vacuum(gradients(Antivector))).trace(1, 2)              # [...] Antivector-


def worldline(orbit_radius: float, orbit_speed: float, times: Scalar) -> tuple[Antivector, Antivector, Antivector]:
    """A charge circling at `orbit_speed` on `orbit_radius` traces worldline positions, lab-time
    tangents, and lab-time bends (`[...] Antivector-` each)."""
    angular_rate = orbit_speed / orbit_radius
    # Turn the starting radius in the `xy` plane, then differentiate by taking commutators with the turning plane.
    turn = (mv.xy * (-0.5 * angular_rate) * times).exp()                     # [...] Rotor+
    offset = turn >> (mv.x.dual() * orbit_radius)                            # [...] Antivector-
    tangent = time + angular_rate * offset.commutator(mv.xy)                 # [...] Antivector-
    bend = angular_rate * (tangent - time).commutator(mv.xy)                 # [...] Antivector-
    return times * time + offset, tangent, bend


def light_cone_gap(
    orbit_radius: float,
    orbit_speed: float,
    events: Antivector,
    times: Scalar,
) -> tuple[Scalar, Scalar, Scalar]:
    """Light emitted from the orbit at `times` has a time shortfall before reaching `events`, a
    shortfall rate with respect to emission time, and a spatial reach (`[...] Scalar+` each)."""
    position, tangent, _ = worldline(orbit_radius, orbit_speed, times)       # [...] Antivector-, Antivector-, Antivector-
    ray = events - position                                                  # [...] Antivector-
    spatial_ray = ray - (mv.t & ray) * time                                  # [...] Antivector-
    distance = (spatial_ray | spatial_ray).square_root()                     # [...] Scalar+
    return (mv.t & events) - times - distance, 1 - (spatial_ray | tangent) / distance, distance


def retarded_time(orbit_radius: float, orbit_speed: float, events: Antivector) -> Scalar:
    """Fixed-point light-cone steps followed by Newton steps find the lab emission time (`[...] Scalar+`)
    whose forward light cone hits each event."""
    times = mv.t & events                                                    # [...] Scalar+
    for _ in range(n_settle_steps):
        _, _, distance = light_cone_gap(orbit_radius, orbit_speed, events, times)
        times = (mv.t & events) - distance                                   # [...] Scalar+
    for _ in range(n_newton_steps):
        shortfall, shortfall_rate, _ = light_cone_gap(orbit_radius, orbit_speed, events, times)
        times = times + shortfall / shortfall_rate                           # [...] Scalar+
    return times


def orbit_potential_gradient(
    orbit_charge: float,
    orbit_radius: float,
    orbit_speed: float,
    events: Antivector,
) -> PotentialGradient:
    """Differentiating the retarded potential of a circling charge with respect to the observation event
    gives the potential gradient (`[...] Vector+ <- Antivector-`)."""
    emission_times = retarded_time(orbit_radius, orbit_speed, events)        # [...] Scalar+
    position, tangent, bend = worldline(orbit_radius, orbit_speed, emission_times)  # [...] Antivector-, Antivector-, Antivector-
    # Scale lab-time derivatives by the Lorentz factor to obtain proper four-velocity and proper four-acceleration.
    dilation = 1 / np.sqrt(1 - orbit_speed**2)
    velocity, acceleration = dilation * tangent, dilation**2 * bend          # [...] Antivector-, Antivector-
    ray = events - position                                                  # [...] Antivector-
    # The charge's instantaneous spatial slice `-velocity.dual()` pairs with `velocity` to 1 and turns with `-acceleration.dual()`.
    rest_slice, slice_rate = -velocity.dual(), -acceleration.dual()          # [...] Vector+, Vector+
    rest_distance = rest_slice & ray                                         # [...] Scalar+
    # Stepping the event shifts the retarded emission time along the null ray tangent `-ray.dual()`.
    emission_slope = (-ray.dual() & Antivector) / rest_distance              # [...] Scalar+ <- Antivector-
    distance_slope = (rest_slice & Antivector) + ((slice_rate & ray) - 1) * emission_slope  # [...] Scalar+ <- Antivector-
    weight = orbit_charge / (4 * np.pi) / rest_distance                      # [...] Scalar+
    return -weight * (slice_rate * emission_slope - rest_slice * distance_slope / rest_distance)  # [...] Vector+ <- Antivector-


# WGSL shader evaluating the circling charge's retarded field at every supersampled pixel on the GPU. Each rank-1 term
# `vector * (form & step)` in `orbit_potential_gradient` has exterior derivative `form ^ vector`, matching the trace of
# `Vector ^ gradients(Antivector)`.
orbit_field_source = f'''
const time = -mv.t.dual();
const n_settle_steps = {n_settle_steps}u;
const n_newton_steps = {n_newton_steps}u;
const pi = {np.pi!r};
var<uniform> orbit_charge: f32;
var<uniform> orbit_radius: f32;
var<uniform> orbit_speed: f32;

struct Worldline {{
    position: Antivector,
    tangent: Antivector,
    bend: Antivector,
}}

fn worldline(lab_time: f32) -> Worldline {{
    let angular_rate = orbit_speed / orbit_radius;
    let turn = (mv.xy * (-0.5 * angular_rate * lab_time)).exp();
    let offset = turn >> (mv.x.dual() * orbit_radius);
    let tangent = time + angular_rate * offset.commutator(mv.xy);
    let bend = angular_rate * (tangent - time).commutator(mv.xy);
    return Worldline(lab_time * time + offset, tangent, bend);
}}

struct LightConeGap {{
    shortfall: f32,
    rate: f32,
    reach: f32,
}}

fn light_cone_gap(event: Antivector, lab_time: f32) -> LightConeGap {{
    let source = worldline(lab_time);
    let ray = event - source.position;
    let spatial_ray = ray - (mv.t & ray) * time;
    let reach = (spatial_ray | spatial_ray).square_root();
    return LightConeGap((mv.t & event) - lab_time - reach, 1 - (spatial_ray | source.tangent) / reach, reach);
}}

fn retarded_time(event: Antivector) -> f32 {{
    var emission_time = mv.t & event;
    for (var index = 0u; index < n_settle_steps; index++) {{
        emission_time = (mv.t & event) - light_cone_gap(event, emission_time).reach;
    }}
    for (var index = 0u; index < n_newton_steps; index++) {{
        let gap = light_cone_gap(event, emission_time);
        emission_time = emission_time + gap.shortfall / gap.rate;
    }}
    return emission_time;
}}

fn orbit_field(event: Antivector) -> Bivector {{
    let source = worldline(retarded_time(event));
    let dilation = 1 / sqrt(1 - orbit_speed * orbit_speed);
    let velocity = dilation * source.tangent;
    let acceleration = dilation * dilation * source.bend;
    let ray = event - source.position;
    let rest_slice = -velocity.dual();
    let slice_rate = -acceleration.dual();
    let rest_distance = rest_slice & ray;
    let emission_form = -ray.dual() / rest_distance;
    let distance_form = rest_slice + ((slice_rate & ray) - 1) * emission_form;
    let weight = orbit_charge / (4 * pi) / rest_distance;
    return -weight * ((emission_form ^ slice_rate) - (distance_form ^ rest_slice) / rest_distance);
}}
'''


# --- plumbing -------------------------------------------------------------------------
def event_grid(half_width: float, half_height: float, n_columns: int, n_rows: int) -> Antivector:
    """Cell-centre spacetime events at lab time zero span `[-half_width, half_width]` by
    `[-half_height, half_height]` in the `z = 0` plane (`[n_rows, n_columns] Antivector-`)."""
    x = (np.arange(n_columns) + 0.5) / n_columns * 2 * half_width - half_width
    y = (np.arange(n_rows) + 0.5) / n_rows * 2 * half_height - half_height
    return (mv.x.dual() * x[None, :] + mv.y.dual() * y[:, None]).cast(Antivector)  # [n_rows, n_columns] Antivector-
