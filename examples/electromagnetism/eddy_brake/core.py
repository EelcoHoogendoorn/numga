"""Currents and braking torque in a conducting disc under a prescribed magnetic field."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from numga import stack
from numga.sparse import SparseExtensor, spdiag
from examples.mesh import Bivector, Mesh, Scalar, Vector, as_scalar, context, ga

mv = context.multivector
Conductivity = ga.gatype((Vector, Vector))
Rotor = ga.gatype.rotor()
MAGNETIC_BLOCK_FACES = 64
HEAT_BLOCK_STEPS = 4


# --- math -----------------------------------------------------------------------------
@dataclass
class Response:
    """The potential, edge circulation and flux, and their mechanical and thermal response."""

    potential: Scalar                    # Scalar[V]
    voltage: Scalar                      # Scalar[E]
    flux: Scalar                         # Scalar[E]
    current: Vector                      # Vector[F]
    heating: Scalar                      # Scalar[F]
    torque: Bivector                      # [] Bivector

    def scaled(self, scale: Scalar) -> Response:
        """The response at a multiple of the reference spin; heating scales quadratically."""
        return Response(self.potential * scale, self.voltage * scale,
                        self.flux * scale, self.current * scale,
                        self.heating * scale.squared(), self.torque * scale)


@dataclass
class Motion:
    """The disc's orientation, spin, accumulated heat, and instantaneous electrical response."""

    orientation: Rotor                   # [] Rotor
    spin: Bivector                       # [] Bivector
    heat: Scalar                         # [] Scalar
    response: Response


@dataclass
class ThermalMotion:
    """A rotating material mesh with accumulated Joule energy on each face."""

    orientation: Rotor                   # [cases] Rotor
    spin: Bivector                       # [cases] Bivector
    vertices: Vector                     # [cases] Vector[V], world positions
    heat: Scalar                         # [cases] Scalar[F], joules per material face


@dataclass
class InductiveMotion(ThermalMotion):
    """Material heat and current, including the energy stored in the current's magnetic field."""

    current: Vector                      # [cases] Vector[F], body-frame sheet current
    streamfunction: Scalar               # [cases] Scalar[V]
    magnetic_energy: Scalar              # [cases] Scalar, joules
    thermal_energy: Scalar               # [cases] Scalar, accumulated modal dissipation
    motor_work: Scalar                   # [cases] Scalar, energy supplied by the motor


@dataclass
class PeriodicDrive:
    """The magnetic drive projected onto current modes over one revolution."""

    mean: Scalar                         # [cases, modes]
    cosine: Scalar                       # [cases, harmonics, modes]
    sine: Scalar                         # [cases, harmonics, modes]
    harmonics: np.ndarray

    def __call__(self, angle: Scalar) -> Scalar:
        phase = angle[..., None] * self.harmonics
        return self.mean + (phase.cos()[..., None] * self.cosine
                            + phase.sin()[..., None] * self.sine).sum(axis=-2)


@dataclass
class CurrentModes:
    """Closed current patterns, normalized by inductance, and their resistive decay rates."""

    decay: Scalar                        # [cases, modes] Scalar, inverse seconds
    currents: Vector                     # [cases, modes] Vector[F]
    streamfunctions: Scalar              # [cases, modes] Scalar[V]
    resistivity: Conductivity            # [cases] Conductivity[F], sheet resistance

    def state(self, mesh: Mesh, orientation: Rotor, spin: Bivector,
              heat: Scalar, amplitudes: Scalar, thermal_energy: Scalar,
              motor_work: Scalar) -> InductiveMotion:
        """Reconstruct the moving current and heat fields from their material amplitudes."""
        current = (self.currents * amplitudes).sum(axis=-1)                # [cases] Vector[F]
        streamfunction = (self.streamfunctions * amplitudes).sum(axis=-1)  # [cases] Scalar[V]
        vertices = orientation >> mesh.vertices                           # [cases] Vector[V]
        magnetic_energy = amplitudes.squared().sum(axis=-1) / 2           # [cases] Scalar
        return InductiveMotion(orientation, spin, vertices, heat, current, streamfunction,
                                magnetic_energy, thermal_energy, motor_work)

    def deposited_heat(self, mesh: Mesh, factors: Scalar) -> Scalar:
        """Read out spatial heat from weighted modal amplitudes `[substeps, cases, modes]`."""
        heat = (mesh.triangle_areas * 0).broadcast_to(self.resistivity.shape)
        # The factors retain all mode-pair products, without storing a dense covariance.
        # Small batches bound the temporary face fields independently of the substep count.
        for start in range(0, len(factors), HEAT_BLOCK_STEPS):
            amplitudes = factors[start:start + HEAT_BLOCK_STEPS]
            current = (self.currents * amplitudes).sum(axis=-1)
            heat = heat + mesh.triangle_areas * (current | self.resistivity(current)).sum(axis=0)
        return heat


def current_modes(mesh: Mesh, conductivity: Conductivity,
                  permeability: float, count: int) -> CurrentModes:
    """The slowest inductive current patterns of a conducting disc, with an insulating rim."""
    boundary = np.unique(mesh.edges[mesh.boundary_edges])
    vertex_count = len(mesh.vertices.batch())
    interior = np.setdiff1d(np.arange(vertex_count), boundary)
    # A streamfunction is zero on the rim. Its turned gradient forms closed currents,
    # with continuous normal flux across every interior edge.
    embedder = SparseExtensor.from_indices(mv.scalar(np.ones((len(interior), 1))),
                                           interior, np.arange(len(interior)),
                                           (vertex_count, len(interior))) * Scalar
    gradient = mesh.reconstruction((mesh.d0 * Scalar)(embedder))           # [F] Vector <- [interior] Scalar
    current = spdiag(mesh.face_planes | Vector)(gradient)                 # [F] Vector <- [interior] Scalar
    # The material map has no normal conduction. Its least-squares inverse gives
    # the resistance acting on tangential currents without inventing a normal component.
    # Each face's map is inverted on its own, as a batch of independent maps.
    resistivity = conductivity.batch().lstsq(Vector).field()              # [cases] Conductivity[F]
    resistance = current.adjoint()(spdiag(mesh.triangle_areas * resistivity)(current))  # [cases, interior] Scalar <- [interior] Scalar
    # Pull each face block back immediately, keeping only the smaller streamfunction map.
    blocks = magnetic_coupling_blocks(mesh, permeability, MAGNETIC_BLOCK_FACES)
    current_adjoint = current.adjoint()
    inductance = current_adjoint(next(blocks)(current))                  # [interior] Scalar <- [interior] Scalar
    for coupling in blocks:
        inductance = inductance + current_adjoint(coupling(current))
    inductance = (inductance + inductance.adjoint()) * 0.5
    # The eigenfields are orthonormal in magnetic energy; their eigenvalues are decay rates.
    decay, modes = resistance.eigh(inductance, count)                     # [cases, modes] Scalar; [cases, modes] Scalar[interior]
    return CurrentModes(decay, current(modes), embedder(modes), resistivity)


def field(positions: Vector, centre: Vector, plane: Bivector,
          strength: float, width: float) -> Bivector:
    """A magnetic footprint whose strength falls as a Gaussian in distance from its centre."""
    distance_squared = (positions - centre).scalar_norm_squared()          # Scalar[E]
    return plane * strength * (-distance_squared / (2 * width ** 2)).exp()


def periodic_drive(mesh: Mesh, modes: CurrentModes, centre: Vector, plane: Bivector,
                   strength: float, width: float, orientation: Rotor, count: int) -> PeriodicDrive:
    """Project one revolution of the magnetic forcing into real harmonic mode coefficients."""
    samples = 2 * count + 1
    angles = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    turns = (plane * (-angles / 2)).exp()
    orientations = turns[:, None] * orientation                          # [samples, cases] Rotor
    positions = orientations >> mesh.face_centers                        # [samples, cases] Vector[F]
    magnetic = orientations << field(positions, centre, plane, strength, width)
    body_plane = orientation << plane                                   # [cases] Bivector
    velocity = -(body_plane | mesh.face_centers)                          # [cases] Vector[F]
    electric = magnetic | velocity                                      # [samples, cases] Vector[F]
    # Project one angle at a time, avoiding a samples × modes × faces temporary.
    values = stack([(modes.currents | (mesh.triangle_areas * force)[..., None]).batch().sum(axis=-1)
                    for force in electric], axis=-2)                    # [cases, samples, modes] Scalar
    harmonics = np.arange(1, count + 1)
    phase = harmonics[:, None] * angles
    cosine = (values[..., None, :, :] * (2 / samples * np.cos(phase))[..., None]).sum(axis=-2)
    sine = (values[..., None, :, :] * (2 / samples * np.sin(phase))[..., None]).sum(axis=-2)
    return PeriodicDrive(values.mean(axis=-2), cosine, sine, harmonics)


def solve(mesh: Mesh, magnetic: Bivector, spin: Bivector,
          conductivity: Conductivity) -> Response:
    """Solve the electric potential that makes the induced currents conserve charge.

    The disc has an insulating boundary. Its magnetic field and angular velocity are prescribed;
    the magnetic field generated by the induced currents is neglected.
    """
    velocity = -(spin | mesh.edge_midpoints)                               # Vector[E]
    lorentz = magnetic | Vector                                            # Vector[E] <- Vector
    # Integrate the motion's electrical drive along each oriented edge.
    emf = lorentz(velocity) | mesh.edge_vectors                             # Scalar[E]

    # Face-averaged Whitney vectors reconstruct edge circulations at each triangle's centroid,
    # where the conductivity map acts on the electrical drive.
    reconstruction = mesh.reconstruction                                   # [F] Vector <- [E] Scalar
    derivative = mesh.d0 * Scalar                                          # [E] Scalar <- [V] Scalar
    gradient = reconstruction(derivative)                                  # [F] Vector <- [V] Scalar
    drive = reconstruction(emf)                                            # Vector[F]
    material = spdiag(mesh.triangle_areas * conductivity)                   # [F, F] Vector <- Vector
    balance = gradient.adjoint()(material(gradient))                       # [V] Scalar <- [V] Scalar
    # The adjoint sums outgoing currents. No exterior fluxes gives an insulating rim.
    # Set one vertex's potential to zero to fix the free additive constant; this changes
    # no edge voltage or current and is not an electrical contact to the disc.
    reference = as_scalar(np.arange(len(mesh.vertices.batch())) == 0).field()   # Scalar[V]
    gauge = spdiag(balance.diagonal() * reference)                         # [V] Scalar <- [V] Scalar
    potential = (balance + gauge).solve(gradient.adjoint()(material(drive)))  # Scalar[V]
    voltage = emf - derivative(potential)                                  # Scalar[E]
    electric = reconstruction(voltage)                                    # Vector[F]
    current = conductivity(electric)                                      # Vector[F]
    # Applying the reconstruction's adjoint returns the conserved dual-edge fluxes.
    flux = reconstruction.adjoint()(mesh.triangle_areas * current)         # Scalar[E]

    # The same skew map gives the magnetic force. The adjoint pairing makes charge balance
    # equate mechanical power loss with Joule heating.
    force = flux * lorentz(mesh.edge_vectors)                               # Vector[E]
    torque = (mesh.edge_midpoints ^ force).batch().sum(axis=-1)             # [] Bivector
    heating = mesh.triangle_areas * (electric | current)                    # Scalar[F]
    return Response(potential, voltage, flux, current, heating, torque)


def braking(mesh: Mesh, magnetic: Bivector, plane: Bivector,
            conductivity: Conductivity, inertia: float, orientation: Rotor,
            spin: Bivector, step: float, steps: int) -> Iterator[Motion]:
    """A freely turning disc with spatially uniform, body-fixed conductivity.

    The spin lies in the fixed unit plane. Mechanical motion advances in time; the current
    equilibrates instantaneously at each orientation and spin. A predicted midpoint gives
    second-order drag, while midpoint angular momentum makes the heat gain equal the kinetic
    energy loss.
    """
    heat = mv.scalar([0]).broadcast_to(spin.shape)                           # [cases] Scalar
    material = orientation >> conductivity(orientation << Vector)          # [cases] Vector <- Vector
    response = solve(mesh, magnetic, spin, material)
    yield Motion(orientation, spin, heat, response)

    for _ in range(steps):
        # Predict where the body fibres point halfway through the step, then solve the
        # current at unit spin. Linearity supplies the response at any spin, including rest.
        midpoint_orientation = (spin * (-step / 4)).exp() * orientation     # [cases] Rotor
        material = midpoint_orientation >> conductivity(midpoint_orientation << Vector)  # [cases] Vector <- Vector
        unit_response = solve(mesh, magnetic, plane, material)
        drag = plane | unit_response.torque                                # [cases] Scalar
        midpoint_spin = spin / (1 + drag * step / (2 * inertia))             # [cases] Bivector

        # The torque uses the mean spin of the step. Its work is exactly the change in
        # rotational kinetic energy; the same mean spin turns the body and heats the metal.
        orientation = (midpoint_spin * (-step / 2)).exp() * orientation      # [cases] Rotor
        spin = 2 * midpoint_spin - spin                                    # [cases] Bivector
        heat = heat + step * midpoint_spin.scalar_norm_squared() * unit_response.heating.batch().sum(axis=-1)
        material = orientation >> conductivity(orientation << Vector)      # [cases] Vector <- Vector
        response = solve(mesh, magnetic, spin, material)
        yield Motion(orientation, spin, heat, response)


def stationary_braking(response: Response, plane: Bivector, inertia: float,
                       spin: Bivector, step: float, steps: int) -> Iterator[Motion]:
    """Free rotation when the material field is invariant under rotation about the disc's axis.

    The supplied response is solved at unit spin in the fixed unit plane. Isotropic, radial and
    circumferential material fields remain fixed in space, so changing speed scales their currents
    without changing their paths. Midpoint drag converts kinetic energy into accumulated heat.
    """
    drag = plane | response.torque                                        # [cases] Scalar
    power = response.heating.batch().sum(axis=-1)                         # [cases] Scalar
    spin = spin.broadcast_to(drag.shape)                                   # [cases] Bivector
    orientation = mv.rotor().broadcast_to(drag.shape)                       # [cases] Rotor
    heat = mv.scalar([0]).broadcast_to(drag.shape)                           # [cases] Scalar
    yield Motion(orientation, spin, heat, response.scaled(-(plane | spin)))

    for _ in range(steps):
        # Unit-spin drag stays fixed; only the mean spin and its turn change each step.
        midpoint_spin = spin / (1 + drag * step / (2 * inertia))             # [cases] Bivector
        orientation = (midpoint_spin * (-step / 2)).exp() * orientation      # [cases] Rotor
        spin = 2 * midpoint_spin - spin                                    # [cases] Bivector
        heat = heat + step * midpoint_spin.scalar_norm_squared() * power    # [cases] Scalar
        yield Motion(orientation, spin, heat, response.scaled(-(plane | spin)))


def material_braking(mesh: Mesh, centre: Vector, plane: Bivector,
                     strength: float, width: float, conductivity: Conductivity,
                     inertia: float, orientation: Rotor, spin: Bivector,
                     heat: Scalar, step: float, steps: int) -> Iterator[ThermalMotion]:
    """Turn a material mesh and accumulate the generated heat on its faces.

    Conductivity and heat are `[cases]` fields over the reference mesh's faces; orientation and
    spin have shape `[cases]`. The unit spin plane and magnetic patch are fixed in world space.
    Heat stays with the material, without thermal diffusion or cooling. The electrical
    response equilibrates instantaneously on the moving domain.
    """
    vertices = orientation >> mesh.vertices                                # [cases] Vector[V]
    yield ThermalMotion(orientation, spin, vertices, heat)

    for _ in range(steps):
        midpoint_orientation = (spin * (-step / 4)).exp() * orientation     # [cases] Rotor
        # Sample the fixed magnet on the moving edges, then pull its field back into
        # the material frame. Incidence, reconstruction and conductivity stay on that mesh.
        positions = midpoint_orientation >> mesh.edge_midpoints            # [cases] Vector[E]
        magnetic = field(positions, centre, plane, strength, width)         # [cases] Bivector[E]
        body_magnetic = midpoint_orientation << magnetic                   # [cases] Bivector[E]
        body_plane = midpoint_orientation << plane                        # [cases] Bivector
        unit_response = solve(mesh, body_magnetic, body_plane, conductivity)
        drag = body_plane | unit_response.torque                          # [cases] Scalar

        # The same midpoint spin sets the mechanical loss and every face's heat gain.
        # Face identities do not change as the mesh turns, so transport needs no resampling.
        midpoint_spin = spin / (1 + drag * step / (2 * inertia))            # [cases] Bivector
        heat = heat + step * midpoint_spin.scalar_norm_squared() * unit_response.heating
        orientation = (midpoint_spin * (-step / 2)).exp() * orientation      # [cases] Rotor
        spin = 2 * midpoint_spin - spin                                    # [cases] Bivector
        vertices = orientation >> mesh.vertices                            # [cases] Vector[V]
        yield ThermalMotion(orientation, spin, vertices, heat)


def inductive_braking(mesh: Mesh, modes: CurrentModes, forcing: PeriodicDrive, plane: Bivector,
                      inertia: float,
                      orientation: Rotor, spin: Bivector, heat: Scalar, amplitudes: Scalar,
                      motor_torque: Bivector, step: float) -> Iterator[InductiveMotion]:
    """Advance current memory, rotation and material heat with coupled midpoint steps.

    Current amplitudes have shape `[cases, modes]` in the material frame. Their magnetic
    energy plus rotational energy and heat increase by the motor's work. Motor torque has
    shape `[steps, substeps, cases]`. Substeps resolve electrical relaxation between frames. Thermal diffusion,
    displacement current and variation through the sheet thickness are omitted.
    """
    substeps = motor_torque.shape[1]
    interval = step / substeps
    angle = mv.scalar([0]).broadcast_to(orientation.shape)                # [cases] Scalar
    thermal_energy = heat.batch().sum(axis=-1)                           # [cases] Scalar
    motor_work = thermal_energy * 0                                      # [cases] Scalar
    factors = mv.scalar(np.zeros((substeps,) + amplitudes.shape + (1,)))  # [substeps, cases, modes] Scalar
    damping = 1 + interval * modes.decay / 2                              # [cases, modes] Scalar
    yield modes.state(mesh, orientation, spin, heat, amplitudes, thermal_energy, motor_work)
    for torques in motor_torque:
        for substep, torque in enumerate(torques):
            speed = -(plane | spin)                                           # [cases] Scalar
            motor = -(plane | torque)                                         # [cases] Scalar
            drive = forcing(angle + speed * (interval / 2))                   # [cases, modes] Scalar
            free_amplitudes = amplitudes / damping                            # [cases, modes] Scalar
            driven_amplitudes = interval / 2 * drive / damping                 # [cases, modes] Scalar
            # Eliminate the midpoint currents from angular momentum, then recover
            # both states together. This includes energy returning from the magnetic field.
            midpoint_speed = (speed + interval / (2 * inertia) * (motor - (drive * free_amplitudes).sum(axis=-1))) / (
                1 + interval / (2 * inertia) * (drive * driven_amplitudes).sum(axis=-1))  # [cases] Scalar
            midpoint_amplitudes = free_amplitudes + midpoint_speed[..., None] * driven_amplitudes  # [cases, modes] Scalar
            # Modal damping gives total heat exactly. Keep weighted amplitudes to
            # reconstruct its location, including cross terms, at the display frame.
            factors = factors.at[substep].set(np.sqrt(interval) * midpoint_amplitudes)
            thermal_energy = thermal_energy + interval * (modes.decay * midpoint_amplitudes.squared()).sum(axis=-1)
            motor_work = motor_work + interval * motor * midpoint_speed
            amplitudes = 2 * midpoint_amplitudes - amplitudes                   # [cases, modes] Scalar
            midpoint_spin = plane * midpoint_speed                            # [cases] Bivector
            angle = angle + interval * midpoint_speed                         # [cases] Scalar
            orientation = (midpoint_spin * (-interval / 2)).exp() * orientation  # [cases] Rotor
            spin = 2 * midpoint_spin - spin                                    # [cases] Bivector
        heat = heat + modes.deposited_heat(mesh, factors)                     # [cases] Scalar[F]
        yield modes.state(mesh, orientation, spin, heat, amplitudes, thermal_energy, motor_work)


# --- plumbing -------------------------------------------------------------------------
def magnetic_coupling(mesh: Mesh, permeability: float) -> SparseExtensor:
    """Integrated magnetic coupling between face currents on a planar thin sheet.

    Each source triangle's Coulomb potential is integrated analytically; three positive
    quadrature points integrate the target triangle. Symmetry enforces magnetic reciprocity,
    and the finite self interaction is integrated exactly. Thickness enters the sheet
    conductivity, not a numerical cutoff in the magnetic kernel.
    """
    coupling = next(magnetic_coupling_blocks(mesh, permeability, len(mesh.faces)))
    return (coupling + coupling.adjoint()) * 0.5


def magnetic_coupling_blocks(mesh: Mesh, permeability: float,
                             block_faces: int) -> Iterator[SparseExtensor]:
    """Target-face blocks of the planar Coulomb integral, before reciprocal symmetrization.

    Summing their pullbacks and then taking the symmetric part gives the same inductance
    as the full face coupling, without keeping all face-pair vector maps in memory.
    """
    corners = mesh.corners                                                  # [corners] Vector[F]
    edges = corners[[1, 2, 0]] - corners                                   # [edges] Vector[F]
    lengths = edges.norm()                                                # [edges] Scalar[F]
    outward = mesh.face_planes | (edges / lengths)                         # [edges] Vector[F]
    # A triangle's double integral stays finite despite the point kernel's singularity.
    semiperimeter = lengths.sum(axis=0) / 2                                # Scalar[F]
    logarithms = (semiperimeter / (semiperimeter - lengths)).log()
    self_integral = (4 / 3) * mesh.triangle_areas.squared() * (logarithms / lengths).sum(axis=0)
    indices = np.arange(len(mesh.faces))
    areas, centres = mesh.triangle_areas.batch(), mesh.face_centers.batch()
    for start in range(0, len(indices), block_faces):
        rows = indices[start:start + block_faces]
        # These positive quadrature points have barycentric weights two-thirds, one-sixth,
        # one-sixth; their equal weights integrate a quadratic exactly.
        targets = (corners.batch()[:, rows] + centres[rows]) / 2           # [samples, block] Vector
        # Target points are a batch against the source faces' field: every pair of the two.
        relative = corners - targets[..., None]                           # [samples, block, edges] Vector[F]
        distances = relative.norm()                                      # [samples, block, edges] Scalar[F]
        endpoint_sum = distances + distances[..., [1, 2, 0]]               # [samples, block, edges] Scalar[F]
        edge_potential = ((endpoint_sum + lengths) / (endpoint_sum - lengths)).log()
        heights = relative | outward                                    # [samples, block, edges] Scalar[F]
        source_potential = (heights * edge_potential).sum(axis=-1)         # [samples, block] Scalar[F]
        integral = (areas[rows] * source_potential.mean(axis=0)).batch()   # [block, F] Scalar
        integral = integral.at[np.arange(len(rows)), rows].set(self_integral.batch()[rows])
        yield SparseExtensor.from_indices(
            integral * (permeability / (4 * np.pi)) * Vector,
            rows[:, None], indices[None, :], (len(indices), len(indices)))  # [F, F] Vector <- Vector
