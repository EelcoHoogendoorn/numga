"""Constitutive maps in spacetime geometric algebra determine wave propagation, reflection, and time-domain transport across stratified optical layers.

Each material maps the electromagnetic field bivector to its excitation antibivector through a sum of dyads. Temporal worldsheet planes swept out by the spatial axes read the electric field and carry the principal permittivities, while spatial plane patches read the magnetic field and carry the inverse permeability. Taking tangential traces at an interface packs the continuous components of field and excitation into the vector and trivector grades of a four-component boundary state. Solving the two derivative-free longitudinal Maxwell constraints alongside those traces reconstructs the full six-component field bivector from the boundary state, and projecting the tangential Maxwell derivative onto the same boundary space gives the layer's propagation generator. Both transverse polarizations stay coupled when a birefringent crystal's optic axis twists from slice to slice or when an axion coupling jumps across a face. Pairing the stress-energy current along the observer's time direction with the interface normal measures the cycle-averaged optical power crossing the face.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from functools import reduce
from itertools import accumulate

import numpy as np

from numga import Algebra, NumpyContext, Orientation, dim, stack
from numga.algebras import STA as ga
from numga.sparse import Sparse

mv = NumpyContext(ga, dtype=np.complex128, units=True).multivector
# Real spacetime context for physical parameters and stepping real fields along z without harmonic phase factors.
real = NumpyContext(ga, units=True).multivector

force_dim = dim.mass * dim.length / dim.time**2
permeability_dim = force_dim / dim.current**2
electric_dim = force_dim / (dim.current * dim.time)                               # mass*length/(current*time^3)
# Coherent vacuum constants keep numerical values O(1) while tracking all electromagnetic dimensions.
c = real.scalar(1.0, unit=dim.length / dim.time)                                  # [] Scalar+, length/time
mu_0 = real.scalar(1.0, unit=permeability_dim)                                    # [] Scalar+, length*mass/(current^2*time^2)
eps_0 = 1 / (mu_0 * c**2)                                                         # [] Scalar+, current^2*time^4/(length^3*mass)
Z_0 = mu_0 * c                                                                    # [] Scalar+, length^2*mass/(current^2*time^3)
admittance_dim = 1 / Z_0.unit                                                     # current^2*time^3/(length^2*mass)

Scalar = ga.gatype.scalar()                                                       # Scalar+
Wavevector = ga.gatype.vector()                                                   # Vector+
Field = ga.gatype.bivector()                                                      # Bivector+
Excitation = ga.gatype.antibivector()                                             # Antibivector-
# Spatial directions, crystal optic axes, displacements along a line, and energy currents are intrinsic antivectors.
Direction = ga.gatype.antivector()                                                # Antivector-
# Tangential electric planes xt and yt spanned by the transverse axes and the observer's time 1-form.
Polarization = Field.derive.touching(Algebra("x+y+"), Algebra("t-"))              # Bivector+
Boundary = ga.gatype(ga.subspace("x y txz tyz"), orientation=Orientation.extrinsic)  # Odd+
Constitutive = ga.gatype((Excitation, Field))                                     # Excitation- <- Field+
Inverse = ga.gatype((Field, Excitation))                                          # Field+ <- Excitation-
FieldMap = ga.gatype((Field, Boundary))                                           # Field+ <- Boundary+
Propagation = ga.gatype((Boundary, Boundary))                                     # Boundary+ <- Boundary+
PolarizationMap = ga.gatype((Polarization, Polarization))                         # Polarization+ <- Polarization+
WaveMap = ga.gatype((Field, Polarization))                                        # Field+ <- Polarization+
Outgoing = ga.gatype((Boundary, Polarization))                                    # Boundary+ <- Polarization+

# Observer rest-frame 4-velocity, spatial principal axes, and spatial magnetic bivector planes.
time = -mv.t.dual()                                                               # Direction-
axes = stack((mv.x.dual(), mv.y.dual(), mv.z.dual()))                             # [3] Direction-
magnetic_planes = stack((mv.yz, mv.zx, mv.xy))                                    # [3] Field+
# Intrinsic worldsheet planes reading the electric field and spatial patches reading the magnetic field.
swept_planes = axes & time                                                        # [3] Excitation-
patches = magnetic_planes.dual()                                                  # [3] Excitation-
# Temporal part of a unit-frequency wavevector so phase -(wavevector & place) grows in time.
unit_frequency = -mv.t                                                            # Wavevector+
field_boundary = mv.t ^ (mv.t.inverse() | (mv.z ^ Field))                         # Boundary+ <- Field+
excitation_boundary = mv.t.inverse() | (mv.t ^ (mv.z ^ Excitation).dual())        # Boundary+ <- Excitation-
# Contracting a boundary state with the inverse normal recovers its tangential electric bivector.
electric_trace = mv.z.inverse() | Boundary                                        # Polarization+ <- Boundary+


# --- math -----------------------------------------------------------------------------
def dielectric_response(permittivity: Scalar, reluctivity: Scalar) -> Constitutive:
    """Principal `permittivity` (`[..., 3]`, `eps_0`) and scalar `reluctivity` (`[...]`, `1/mu_0`) assemble the rest-frame constitutive dyad sum (`[...] Excitation- <- Field+`)."""
    # Time squares to minus one, so electric worldsheet dyads enter with a minus sign.
    electric = -(permittivity * swept_planes * (Field & swept_planes)).sum(axis=-1)
    magnetic = (patches * (Field & patches)).sum(axis=-1)
    return electric + reluctivity * magnetic                                      # [...] Excitation- <- Field+


def layer_operators(response: Constitutive, parallel: Wavevector) -> tuple[FieldMap, Propagation]:
    """Reducing Maxwell's equations in a layer with constitutive map `response` (`[...] Excitation- <- Field+`) normal to `z` at tangential slowness `parallel` yields the full-field reconstruction map and the normal propagation generator."""
    # Four tangential combinations of field and excitation stay continuous across interfaces.
    boundary = field_boundary + excitation_boundary(response)                     # [...] Boundary+ <- Field+
    # Two longitudinal Maxwell equations contain no normal derivative and vanish identically.
    constraint = (-(mv.z | (mv.z ^ (parallel ^ Field)))
                  - mv.z * (mv.z | (parallel ^ response).dual()))
    reconstruction = (boundary + constraint).solve(Boundary)                      # [...] Field+ <- Boundary+
    # Contracting parallel ^ F + z ^ dF/dz == 0 with z.inverse() gives the normal derivative of the boundary state.
    generator = -(field_boundary(mv.z.inverse() | (parallel ^ reconstruction))
                  + excitation_boundary(mv.z.inverse() | (parallel ^ response(reconstruction))))
    return reconstruction, generator                                              # [...] Field+ <- Boundary+, Boundary+ <- Boundary+


def layer_propagation(generator: Propagation, travel: Scalar) -> Propagation:
    """Exponentiating `generator` (`[...] Boundary+ <- Boundary+`) over `travel` (`omega * thickness`, `length/time`) in its eigenmodes builds the transfer map carrying boundary states from a layer's exit face back to its entrance face."""
    indices, modes = generator.eig()                                              # [..., 4] Scalar+, Boundary+
    frame = (modes * modes.scalar_product(Boundary)).sum(axis=-1)                 # [...] Boundary+ <- Boundary+
    # Solving the mode-dyad frame supplies the biorthogonal reciprocal reading each mode's amplitude.
    reciprocal = frame[..., None].solve(modes)                                    # [..., 4] Boundary+
    phases = (1j * indices * travel[..., None]).exp()                             # [..., 4] Scalar+
    return (phases * modes * reciprocal.scalar_product(Boundary)).sum(axis=-1)    # [...] Boundary+ <- Boundary+


def stack_propagation(layers: Iterable[Propagation]) -> Propagation:
    """Composing `layers` ordered from incident medium to substrate builds the transfer map of the full stack."""
    return reduce(lambda total, layer: total(layer), layers)                      # [...] Boundary+ <- Boundary+


def exterior_wave(slowness: Scalar, parallel: Wavevector) -> WaveMap:
    """Completing the 4-potential perpendicular to the ray builds the forward exterior plane-wave map (`[...] Field+ <- Polarization+`) at medium `slowness` (`n/c`, `time/length`) and tangential slowness `parallel`."""
    spatial = mv.t.inverse() | (mv.t ^ parallel)
    normal_slowness = (slowness ** 2 - spatial.squared()).square_root()
    # Undo the temporal wedge and add a normal part so wavevector ^ potential has Polarization as its tangential electric trace.
    tangential = unit_frequency.inverse() | Polarization                          # Wavevector+ <- Polarization+
    potential = tangential - mv.z * (parallel | tangential) / normal_slowness
    wavevector = parallel + mv.z * normal_slowness
    return wavevector ^ potential                                                 # [...] Field+ <- Polarization+


def interface_scattering(
    entrance: Outgoing,
    incident_response: Constitutive,
    incident_slowness: Scalar,
    parallel: Wavevector,
) -> tuple[PolarizationMap, PolarizationMap]:
    """Matching an `entrance` boundary map (`[...] Boundary+ <- Polarization+`) against an exterior medium of `incident_response` and `incident_slowness` (`n/c`) at `parallel` solves for the polarization reflection and transmission maps."""
    _, generator = layer_operators(incident_response, parallel)
    spatial = mv.t.inverse() | (mv.t ^ parallel)
    normal_slowness = (incident_slowness ** 2 - spatial.squared()).square_root()
    incoming = electric_trace((Boundary + generator / normal_slowness) / 2)       # [...] Polarization+ <- Boundary+
    returning = electric_trace((Boundary - generator / normal_slowness) / 2)      # [...] Polarization+ <- Boundary+
    transmission = incoming(entrance).inverse()                                   # [...] Polarization+ <- Polarization+
    reflection = returning(entrance)(transmission)                                # [...] Polarization+ <- Polarization+
    return reflection, transmission


def wave_power(field: Field, response: Constitutive) -> Scalar:
    """Pairing the stress-energy current of `field` (`[...] Field+`) in `response` (`[...] Excitation- <- Field+`) with the interface normal `z` sums the optical power (`[...] Scalar+`) over both real quadratures of an oscillation cycle."""
    excitation = response(field)                                                  # [...] Excitation-
    fields = stack([field.real(), (-1j * field).real()])                          # [2, ...] Field+
    excitations = stack([excitation.real(), (-1j * excitation).real()])           # [2, ...] Excitation-
    observer, face = time.real(), mv.z.real()                                     # Direction-, Wavevector+
    currents = (fields & excitations) / 2 * observer - ((fields & observer) ^ excitations)  # [2, ...] Direction-
    return (face & currents).sum(axis=0)                                          # [...] Scalar+


def scattering_powers(
    reflection: PolarizationMap,
    transmission: PolarizationMap,
    polarization: Polarization,
    incident_wave: WaveMap,
    incident_response: Constitutive,
    substrate_wave: WaveMap,
    substrate_response: Constitutive,
) -> tuple[Scalar, Scalar]:
    """Normalizing reflected and transmitted wave powers by the incident wave power returns the reflectance and transmittance (`[...] Scalar+`, `[...] Scalar+`) for `polarization`."""
    incident_power = wave_power(incident_wave(polarization), incident_response)
    reflectance = wave_power(incident_wave(reflection(polarization)), incident_response) / incident_power
    transmittance = wave_power(substrate_wave(transmission(polarization)), substrate_response) / incident_power
    return reflectance, transmittance


def interior_fields(
    reconstruction: FieldMap,
    generator: Propagation,
    thicknesses: Scalar,
    wavelength: Scalar,
    exit_state: Boundary,
    fractions: np.ndarray,
) -> Field:
    """Propagating `exit_state` (`[...] Boundary+`) backward across layers of `thicknesses` (`[n_layers]`) at `wavelength` reconstructs the full field bivectors (`[..., n_layers, n_samples] Field+`) at fractional layer depths `fractions` (`[n_samples]`)."""
    travel = 2 * np.pi * c * thicknesses / wavelength
    crossings = layer_propagation(generator, travel)                              # [...] Boundary+ <- Boundary+
    # Accumulate layer crossings from the substrate back so each layer knows the boundary state at its exit face.
    faces = accumulate(crossings[:0:-1], lambda state, crossing: crossing(state), initial=exit_state)
    right_faces = stack(tuple(faces)[::-1], axis=-1)                              # [...] Boundary+
    inward = layer_propagation(generator[:, None], travel[:, None] * (1 - fractions))  # [...] Boundary+ <- Boundary+
    return reconstruction[:, None](inward(right_faces[..., None]))                # [...] Field+


def leapfrog_states(
    difference: Sparse,
    node_inverse: Inverse,
    edge_response: Constitutive,
    electric: Field,
    magnetic: Field,
    spacing: Scalar,
    interval: Scalar,
    n_steps: int,
) -> Iterator[tuple[Field, Field]]:
    """Yields staggered `(electric, magnetic)` field states (`[...] Field+[V]`, `[...] Field+[E]`) across `n_steps` leapfrog updates of `d ^ F == 0` and `d ^ G == 0` on a 1D periodic loop."""
    rate = interval / spacing
    for _ in range(n_steps):
        # Faraday update d ^ F == 0 advances edge magnetic planes from neighbour electric differences.
        magnetic = magnetic - (real.t.inverse() | (real.z ^ (difference * electric))) * rate
        excitation = edge_response(magnetic)
        # Ampere-Maxwell update d ^ G == 0 pulls edge excitations back onto the nodes via the adjugate difference.
        displacement = (real.t.inverse() | (real.z ^ (difference.adjugate() * excitation))) * rate
        electric = electric + node_inverse(displacement)
        yield electric, magnetic


def electromagnetic_energy(
    electric: Field,
    magnetic: Field,
    node_response: Constitutive,
    edge_response: Constitutive,
    spacing: Scalar,
) -> Scalar:
    """Pairing `electric` (`[...] Field+[V]`) on the nodes and `magnetic` (`[...] Field+[E]`) on the staggered edges with their excitations returns the electromagnetic energy (`[...] Scalar+[V]`) of each staggered interval of length `spacing`."""
    # Spatial magnetic bivectors pair with their excitations to minus twice the magnetic energy density.
    return ((electric & node_response(electric)) - (magnetic & edge_response(magnetic))) * (spacing / 2)
