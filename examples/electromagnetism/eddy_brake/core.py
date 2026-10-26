"""Eddy-current braking: the currents and braking torque in a conducting disc turning under a prescribed
magnetic field.

The disc lives in the spacetime of its plane, `x+y+t-`. Its points and edges are directions, antivectors,
intrinsic like all geometry, and the physics is their pairing with extrinsic quantities. A point moves
under the spin by their commutator, `position.commutator(spin)`. The magnet's field across the disc is the
plane `xy`; the electric field the moving material reads is a plane through time, the magnetic plane met
with the velocity and swept through the observer's slice of time, `(velocity & magnetic) ^ mv.t`. It sits
on the edges: its pairing with an edge swept through time, `electric & (edge & time)`, is the edge's
electromotive force. The current it drives is a direction like the edges; the field read along it,
`electric & (current & time)`, is the heat, and the currents' fluxes through the dual edges are intrinsic
amounts, pairing with the voltages on the edges into power, `voltage & flux`. A force's moment about the
origin, `force | midpoint`, pairs with the spin, `spin & moment`, into the power it feeds the turning."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from numga import stack
from numga.algebras import STA2D
from numga.sparse import SparseExtensor, spdiag
from examples import instantiate

# The spacetime of the disc's plane, space positive.
ga = STA2D
mesh_module = instantiate("examples.util.mesh", ga)
Mesh, Scalar, Point = mesh_module.Mesh, mesh_module.Scalar, mesh_module.Point
Vector, Bivector = ga.gatype.vector(), ga.gatype.bivector()
as_scalar, context = mesh_module.as_scalar, mesh_module.ctx

mv = context.multivector
# The electric field, a plane through time; currents, directions in the disc like its edges.
Field = Bivector.derive.containing("t")
Current = Point.derive.containing("t")
# Fluxes and charges: amounts, intrinsic, paired with voltages and potentials into power.
Amount = ga.gatype.pseudoscalar().derive.flipped_orientation
Conductivity = ga.gatype((Current, Field))                 # Current <- Field
Resistivity = ga.gatype((Field, Current))                  # Field <- Current
Projection = ga.gatype((Scalar, Scalar))
Reading = ga.gatype((Scalar, Amount))
Rotor = ga.gatype.rotor()

# The direction in time of an observer at rest with the magnet, which pairs to one with `mv.t`; the dual
# carries the minus of t's square.
time = -mv.t.dual()                                        # [] Antivector
# A quarter turn in the disc, from x towards y.
quarter_turn = (mv.xy * (-np.pi / 4)).exp()
# A force's moment about the origin: the force met with the point it acts at, intrinsic.
Moment = Vector.derive.flipped_orientation
n_magnetic_block_faces = 64


# --- math -----------------------------------------------------------------------------
@dataclass
class Disc:
    """A disc's mesh, its vertices, and the geometry they give it."""

    mesh: Mesh
    vertices: Point                      # Point[V]
    edge_displacements: Point            # Point[E]
    edge_midpoints: Point                # Point[E]
    corners: Point                       # [3] Point[F]
    triangle_edges: Point                # [3] Point[F]
    triangle_areas: Scalar               # Scalar[F]
    face_centers: Point                  # Point[F]
    reconstruction: SparseExtensor       # [F, E] Vector <- Scalar


def disc(mesh: Mesh, vertices: Point) -> Disc:
    """The geometry of a disc's mesh at its vertices `Point[V]`: each edge from its tail to its head, and its
    midpoint; each face's corners, the edge facing each corner, counter-clockwise, the face's area, half the size of
    the plane two of its edges span, and its centre; and Whitney's reconstruction of a field from its edge
    circulations at the face centres."""
    edges = mesh.d0 * vertices                                             # Point[E]
    ends = vertices.batch()[mesh.edges.T]                                  # [2, E] Point
    corners = vertices.batch()[mesh.faces.T].field()                       # [3] Point[F]
    # The edge facing each corner is one of the edges, signed as the face runs along it.
    sides = (edges.batch()[mesh.face_edges.T] * mesh.face_edge_orientation.T).field()   # [3] Point[F]
    areas = sides[1].commutator(sides[0]).norm() / 2                       # Scalar[F]
    # Each corner's hat function has its gradient across the edge facing it, that edge read through the face's
    # normal over twice the area; all three hats are a third at the centroid, and the difference of two gradients
    # reconstructs an edge's circulation, signed as the mesh edge runs.
    normals = (sides[0] & sides[1]) / (2 * areas)                          # Vector[F]
    gradients = (sides | normals.inverse()) / (2 * areas)                  # [3] Vector[F]
    signs = as_scalar(mesh.face_edge_orientation.T).field()                # [3] Scalar[F]
    reconstruction = mesh.by_corner((gradients[[2, 0, 1]] - gradients[[1, 2, 0]]) / 3 * signs * Scalar)
    return Disc(mesh, vertices, edges, ((ends[0] + ends[1]) / 2).field(), corners, sides, areas,
                corners.sum(axis=0) / 3, reconstruction)


@dataclass
class Response:
    """The potential, edge circulation and flux, and their mechanical and thermal response."""

    potential: Scalar                    # Scalar[V]
    voltage: Scalar                      # Scalar[E]
    flux: Amount                         # Amount[E]
    current: Current                     # Current[F]
    heating: Scalar                      # Scalar[F]
    torque: Moment                       # [] Moment


@dataclass
class Motion:
    """A rotating material mesh with its sheet current and the Joule energy accumulated on each face."""

    orientation: Rotor                   # [cases] Rotor
    spin: Bivector                       # [cases] Bivector
    vertices: Point                      # [cases] Point[V], world positions
    heat: Scalar                         # [cases] Scalar[F], joules per material face
    current: Current                     # [cases] Current[F], body-frame sheet current
    streamfunction: Scalar               # [cases] Scalar[V], whose contours are the current's paths


@dataclass
class InductiveMotion(Motion):
    """Material heat and current, including the energy stored in the current's magnetic field."""

    magnetic_energy: Scalar              # [cases] Scalar, joules
    thermal_energy: Scalar               # [cases] Scalar, accumulated modal dissipation
    motor_work: Scalar                   # [cases] Scalar, energy supplied by the motor


@dataclass
class PeriodicDrive:
    """The magnetic drive projected onto current modes over one revolution."""

    mean: Scalar                         # [cases] Scalar[modes]
    cosine: Scalar                       # [cases, harmonics] Scalar[modes]
    sine: Scalar                         # [cases, harmonics] Scalar[modes]
    harmonics: np.ndarray                # [harmonics]

    def __call__(self, angle: Scalar) -> Scalar:
        phase = angle[..., None] * self.harmonics                          # [..., cases, harmonics] Scalar
        return self.mean + (phase.cos() * self.cosine + phase.sin() * self.sine).sum(axis=-1)


@dataclass
class CurrentModes:
    """Closed current patterns, normalized by inductance, and their resistive decay rates."""

    decay: Scalar                        # [cases] Scalar[modes], inverse seconds
    projection: Projection               # [cases] Scalar[modes] <- Scalar[interior]
    reading: Reading                     # [cases] Scalar[modes] <- Amount[interior]
    current: SparseExtensor              # [F] Current <- [interior] Scalar
    embedder: SparseExtensor             # [V] Scalar <- [interior] Scalar
    resistivity: Resistivity             # [cases] Resistivity[F], sheet resistance

    def state(self, disc: Disc, orientation: Rotor, spin: Bivector,
              heat: Scalar, amplitudes: Scalar, thermal_energy: Scalar,
              motor_work: Scalar) -> InductiveMotion:
        """Reconstruct the moving current and heat fields from their material amplitudes."""
        vertices = orientation >> disc.vertices                           # [cases] Point[V]
        magnetic_energy = amplitudes.squared().sites.sum() / 2            # [cases] Scalar
        interior = self.projection.adjoint()(amplitudes)                  # [cases] Scalar[interior]
        return InductiveMotion(orientation, spin, vertices, heat, self.current(interior),
                               self.embedder(interior), magnetic_energy, thermal_energy, motor_work)

    def deposited_heat(self, disc: Disc, factors: Scalar) -> Scalar:
        """Read out spatial heat from weighted modal amplitudes `[substeps, cases] Scalar[modes]`."""
        # The factors retain all mode-pair products, without storing a dense covariance.
        current = self.current(self.projection.adjoint()(factors))        # [substeps, cases] Current[F]
        return disc.triangle_areas * (self.resistivity(current) & (current & time)).sum(axis=0)


def fibre_conductivity(disc: Disc, principal: np.ndarray, inner: float, outer: float,
                       mean: float) -> Conductivity:
    """Sheet conductance of radial and circumferential fibres in an annulus between two radii.

    `principal` `[cases, axes]` holds each material's conductance along the radius and around it;
    outside the annulus, the disc conducts `mean` in every direction.
    """
    distance = disc.face_centers.squared().square_root()                   # Scalar[F]
    # Each face's centre along the radius, and turned a quarter in the disc, around it.
    radial = (disc.face_centers / distance).cast(Current)                  # Current[F]
    directions = stack((radial, quarter_turn >> radial), axis=-1)          # [axes] Current[F]
    fibre_band = as_scalar((distance >= inner) & (distance <= outer)).field()
    local_principal = mean + fibre_band * (principal - mean)               # [cases, axes] Scalar[F]
    # Each dyad reads the field along one direction, its pairing with the plane the direction sweeps through time.
    return (directions * (Field & (directions & time)) * local_principal).sum(axis=-1)   # [cases] Conductivity[F]


def closed_currents(disc: Disc) -> tuple[SparseExtensor, SparseExtensor]:
    """Streamfunctions zero on the rim, from their interior values, and their currents.

    A streamfunction's current runs along each face's edges, each weighted by the streamfunction at the
    corner it faces, over twice the face's area: closed currents, with continuous flux across every
    interior edge.
    """
    boundary = np.unique(disc.mesh.edges[disc.mesh.boundary_edges])
    n_vertices = disc.mesh.n_vertices
    interior = np.setdiff1d(np.arange(n_vertices), boundary)
    embedder = ~SparseExtensor.selection(context, interior, n_vertices) * Scalar  # [V] Scalar <- [interior] Scalar
    edges = disc.triangle_edges.cast(Current)                              # [3] Current[F]
    current = disc.mesh.at_corners(edges / (2 * disc.triangle_areas) * Scalar)(embedder)   # [F] Current <- [interior] Scalar
    return embedder, current


@dataclass
class Streamlines:
    """The streamfunction of the closed current nearest a face current, nearest in the heat their difference makes in
    the material: the current runs along its contours."""

    embedder: SparseExtensor             # [V] Scalar <- [interior] Scalar
    pairing: SparseExtensor              # [cases, interior] Amount <- [F] Current, each closed current's heat with a face current
    resistance: SparseExtensor           # [cases, interior] Amount <- [interior] Scalar

    def __call__(self, current: Current) -> Scalar:
        """The streamfunction nearest a face current `[cases] Current[F]`, `[cases] Scalar[V]`."""
        return self.embedder(self.resistance.solve(self.pairing(current)))


def streamlines(disc: Disc, conductivity: Conductivity) -> Streamlines:
    """The streamfunctions of a material's face currents, from the closed currents of streamfunctions zero on the rim."""
    embedder, current = closed_currents(disc)
    face_ones = as_scalar(np.ones(len(disc.mesh.faces))).field()
    # Each face's conductance, its area times the material's map there, a field whether or not the material varies
    # over the faces, inverted face by face.
    inverse_conductance = (disc.triangle_areas * conductivity).batch().inverse().field()   # [cases] Resistivity[F]
    # The heat each closed current makes with a face current: the closed current swept through time, paired with
    # the field the face current takes in the material, over each face's area.
    pairing = spdiag(face_ones * (Current & time))(current).adjugate()(spdiag(disc.triangle_areas ** 2 * inverse_conductance))
    return Streamlines(embedder, pairing, pairing(current))


def current_modes(disc: Disc, conductivity: Conductivity,
                  permeability: float, n_modes: int) -> CurrentModes:
    """The slowest inductive current patterns of a conducting disc, with an insulating rim."""
    embedder, current = closed_currents(disc)
    face_ones = as_scalar(np.ones(len(disc.mesh.faces))).field()
    # Each face's material map is inverted on its own, as a batch of independent maps.
    resistivity = conductivity.batch().inverse().field()                  # [cases] Resistivity[F]
    # The resistance pairs each current's field with every current swept through time: a form on
    # streamfunctions.
    swept = spdiag(face_ones * (Current & time))(current)                 # [F] Vector <- [interior] Scalar
    resistance = swept.adjugate()(spdiag(disc.triangle_areas * resistivity)(current))  # [cases, interior] Amount <- [interior] Scalar
    # Pull each face block back immediately, keeping only the smaller streamfunction map.
    blocks = magnetic_coupling_blocks(disc.corners, permeability, n_magnetic_block_faces)
    # Each current's vector potential, read along every current through the current map's pairing transpose.
    pairing = current.adjugate()                                         # [interior] Amount <- [F] Vector
    inductance = pairing(next(blocks)(current))                          # [interior] Amount <- [interior] Scalar
    for coupling in blocks:
        inductance = inductance + pairing(coupling(current))
    inductance = (inductance + inductance.adjugate()) / 2
    # The eigenfields are orthonormal in magnetic energy; their eigenvalues are decay rates.
    decay, modes = resistance.eigh(inductance, n_modes)                     # [cases, modes] Scalar; [cases, modes] Scalar[interior]
    # Each pattern's streamfunction on the interior, read as the map onto its amplitude, and paired with
    # amounts on the interior.
    projection = (modes.batch() * Scalar).field(0, 1)                     # [cases] Scalar[modes] <- Scalar[interior]
    reading = (modes.batch() & Amount).field(0, 1)                        # [cases] Scalar[modes] <- Amount[interior]
    return CurrentModes(decay.field(), projection, reading, current, embedder, resistivity)


def periodic_drive(disc: Disc, modes: CurrentModes, centre: Point, plane: Bivector,
                   strength: float, width: float, orientation: Rotor, n_harmonics: int) -> PeriodicDrive:
    """Project one revolution of the magnetic forcing into real harmonic mode coefficients."""
    n_samples = 2 * n_harmonics + 1
    angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    turns = (plane * (-angles / 2)).exp()
    orientations = turns[:, None] * orientation                          # [samples, cases] Rotor
    positions = orientations >> disc.face_centers                        # [samples, cases] Point[F]
    magnetic = orientations << (plane * strength * (-(positions - centre).squared() / (2 * width ** 2)).exp())
    body_plane = orientation << plane                                   # [cases] Bivector
    velocity = disc.face_centers.commutator(body_plane)                  # [cases] Point[F]
    electric = (velocity & magnetic) ^ mv.t                             # [samples, cases] Field[F]
    face_ones = as_scalar(np.ones(len(disc.mesh.faces))).field()
    swept = spdiag(face_ones * (Current & time))(modes.current)          # [F] Vector <- [interior] Scalar
    values = modes.reading(swept.adjugate()(disc.triangle_areas * electric))   # [samples, cases] Scalar[modes]
    harmonics = np.arange(1, n_harmonics + 1)
    phase = angles[:, None] * harmonics                                 # [samples, harmonics]
    cosine = 2 * (values[..., None] * np.cos(phase)[:, None]).mean(axis=0)  # [cases, harmonics] Scalar[modes]
    sine = 2 * (values[..., None] * np.sin(phase)[:, None]).mean(axis=0)    # [cases, harmonics] Scalar[modes]
    return PeriodicDrive(values.mean(axis=0), cosine, sine, harmonics)


def solve(disc: Disc, magnetic: Bivector, spin: Bivector,
          conductivity: Conductivity) -> Response:
    """Solve the electric potential that makes the induced currents conserve charge.

    The disc has an insulating boundary. Its magnetic field and angular velocity are prescribed;
    the magnetic field generated by the induced currents is neglected.
    """
    face_ones = as_scalar(np.ones(len(disc.mesh.faces))).field()
    swept_currents = spdiag(face_ones * (Current & time))                  # [F, F] Vector <- Current
    # Each edge's midpoint moves under the spin, and reads the magnet's field as electric; paired with
    # the edge swept through time, that field is the edge's electromotive force.
    velocity = disc.edge_midpoints.commutator(spin)                        # Point[E]
    emf = ((velocity & magnetic) ^ mv.t) & (disc.edge_displacements & time)   # Scalar[E]

    # Face-averaged Whitney vectors reconstruct edge circulations at each triangle's centroid; swept
    # through time they are the field there, on which the conductivity map acts.
    reconstruction = spdiag(face_ones * (Vector ^ mv.t))(disc.reconstruction)   # [F] Field <- [E] Scalar
    derivative = disc.mesh.d0 * Scalar                                          # [E] Scalar <- [V] Scalar
    gradient = reconstruction(derivative)                                  # [F] Field <- [V] Scalar
    drive = reconstruction(emf)                                            # Field[F]
    material = spdiag(disc.triangle_areas * conductivity)                   # [F, F] Current <- Field
    # The charge each potential's current leaves at each vertex: the field map's pairing transpose,
    # against the currents swept through time.
    outflow = gradient.adjugate()(swept_currents)                          # [V] Amount <- [F] Current
    balance = outflow(material(gradient))                                  # [V] Amount <- [V] Scalar
    # No exterior fluxes gives an insulating rim.
    # Set one vertex's potential to zero to fix the free additive constant; this changes
    # no edge voltage or current and is not an electrical contact to the disc.
    # A least-squares solve of the singular balance finds the same currents at many times the cost.
    reference = as_scalar(np.arange(disc.mesh.n_vertices) == 0).field()   # Scalar[V]
    gauge = spdiag(balance.diagonal() * reference)                         # [V] Scalar <- [V] Scalar
    potential = (balance + gauge).solve(outflow(material(drive)))          # Scalar[V]
    voltage = emf - derivative(potential)                                  # Scalar[E]
    electric = reconstruction(voltage)                                    # Field[F]
    current = conductivity(electric)                                      # Current[F]
    flux = reconstruction.adjugate()(swept_currents(disc.triangle_areas * current))   # Amount[E]

    # Each edge's flux meets the magnetic field's plane with the edge: the magnetic force. The pairing
    # transpose makes charge balance equate mechanical power loss with Joule heating.
    force = flux & (disc.edge_displacements & magnetic)                     # Vector[E]
    # Each force's moment about the origin: the force met with the edge's midpoint. The contraction is the
    # join written once: it equals `midpoint & force.dual()`, the midpoint joined with the force turned into a
    # direction, which takes the dual only to undo it.
    torque = (force | disc.edge_midpoints).sites.sum()                      # [] Moment
    heating = disc.triangle_areas * (electric & (current & time))           # Scalar[F]
    return Response(potential, voltage, flux, current, heating, torque)


def braking(disc: Disc, centre: Point, plane: Bivector,
            strength: float, width: float, conductivity: Conductivity,
            inertia: float, orientation: Rotor, spin: Bivector,
            heat: Scalar, step: float, n_steps: int) -> Iterator[Motion]:
    """Turn a material mesh under a fixed magnet and accumulate the generated heat on its faces.

    Conductivity and heat are `[cases]` fields over the reference mesh's faces; orientation and
    spin have shape `[cases]`. The unit spin plane and magnetic patch are fixed in world space.
    Heat stays with the material, without thermal diffusion or cooling. The electrical
    response equilibrates instantaneously on the moving domain. A predicted midpoint gives
    second-order drag, while midpoint angular momentum makes the heat gain equal the kinetic
    energy loss.
    """
    def response(orientation: Rotor, spin: Bivector) -> Response:
        # Sample the fixed magnet on the moving edges, then pull its field back into
        # the material frame. Incidence, reconstruction and conductivity stay on that mesh.
        positions = orientation >> disc.edge_midpoints                     # [cases] Point[E]
        magnetic = orientation << (plane * strength * (-(positions - centre).squared() / (2 * width ** 2)).exp())  # [cases] Bivector[E]
        return solve(disc, magnetic, orientation << spin, conductivity)

    def state(orientation: Rotor, spin: Bivector, heat: Scalar) -> Motion:
        vertices = orientation >> disc.vertices                            # [cases] Point[V]
        current = response(orientation, spin).current                     # [cases] Current[F]
        return Motion(orientation, spin, vertices, heat, current, paths(current))

    paths = streamlines(disc, conductivity)

    yield state(orientation, spin, heat)
    for _ in range(n_steps):
        # Predict where the material points halfway through the step, then solve the
        # current at unit spin. Linearity supplies the response at any spin, including rest.
        midpoint_orientation = (spin * (-step / 4)).exp() * orientation     # [cases] Rotor
        unit_response = response(midpoint_orientation, plane)
        # The moment paired with the turning plane: the power it takes out, per unit spin squared.
        drag = -((midpoint_orientation << plane) & unit_response.torque)   # [cases] Scalar

        # The same midpoint spin sets the mechanical loss and every face's heat gain.
        # Face identities do not change as the mesh turns, so transport needs no resampling.
        midpoint_spin = spin / (1 + drag * step / (2 * inertia))            # [cases] Bivector
        heat = heat + step * midpoint_spin.scalar_norm_squared() * unit_response.heating
        orientation = (midpoint_spin * (-step / 2)).exp() * orientation     # [cases] Rotor
        spin = 2 * midpoint_spin - spin                                     # [cases] Bivector
        yield state(orientation, spin, heat)


def inductive_braking(disc: Disc, modes: CurrentModes, forcing: PeriodicDrive, plane: Bivector,
                      inertia: float,
                      orientation: Rotor, spin: Bivector, heat: Scalar, amplitudes: Scalar,
                      motor_torque: Scalar, step: float) -> Iterator[InductiveMotion]:
    """Advance current memory, rotation and material heat with coupled midpoint steps.

    Current amplitudes are `[cases] Scalar[modes]` in the material frame. Their magnetic
    energy plus rotational energy and heat increase by the motor's work. Motor torque has
    shape `[steps, substeps, cases]`. Substeps resolve electrical relaxation between frames. Thermal diffusion,
    displacement current and variation through the sheet thickness are omitted.
    """
    n_substeps = motor_torque.shape[1]
    interval = step / n_substeps
    angle = mv.scalar([0]).broadcast_to(orientation.shape)                # [cases] Scalar
    thermal_energy = heat.sites.sum()                                    # [cases] Scalar
    motor_work = thermal_energy * 0                                      # [cases] Scalar
    factors = (amplitudes * 0).broadcast_to((n_substeps,) + amplitudes.shape)  # [substeps, cases] Scalar[modes]
    damping = 1 + interval * modes.decay / 2                              # [cases] Scalar[modes]
    yield modes.state(disc, orientation, spin, heat, amplitudes, thermal_energy, motor_work)
    for torques in motor_torque:
        # The motor's torque about the turning plane, in the plane's own sense.
        for substep, motor in enumerate(torques):
            speed = -(plane | spin)                                           # [cases] Scalar
            drive = forcing(angle + speed * (interval / 2))                   # [cases] Scalar[modes]
            free_amplitudes = amplitudes / damping                            # [cases] Scalar[modes]
            driven_amplitudes = interval / 2 * drive / damping                 # [cases] Scalar[modes]
            # Eliminate the midpoint currents from angular momentum, then recover
            # both states together. This includes energy returning from the magnetic field.
            midpoint_speed = (speed + interval / (2 * inertia) * (motor - (drive * free_amplitudes).sites.sum())) / (
                1 + interval / (2 * inertia) * (drive * driven_amplitudes).sites.sum())  # [cases] Scalar
            midpoint_amplitudes = free_amplitudes + midpoint_speed * driven_amplitudes  # [cases] Scalar[modes]
            # Modal damping gives total heat exactly. Keep weighted amplitudes to
            # reconstruct its location, including cross terms, at the display frame.
            factors = factors.at[substep].set(np.sqrt(interval) * midpoint_amplitudes)
            thermal_energy = thermal_energy + interval * (modes.decay * midpoint_amplitudes.squared()).sites.sum()
            motor_work = motor_work + interval * motor * midpoint_speed
            amplitudes = 2 * midpoint_amplitudes - amplitudes                   # [cases] Scalar[modes]
            angle = angle + interval * midpoint_speed                         # [cases] Scalar
            midpoint_spin = plane * midpoint_speed                            # [cases] Bivector
            orientation = (midpoint_spin * (-interval / 2)).exp() * orientation  # [cases] Rotor
            spin = 2 * midpoint_spin - spin                                    # [cases] Bivector
        heat = heat + modes.deposited_heat(disc, factors)                     # [cases] Scalar[F]
        yield modes.state(disc, orientation, spin, heat, amplitudes, thermal_energy, motor_work)


# --- plumbing -------------------------------------------------------------------------
def magnetic_coupling(corners: Point, permeability: float) -> SparseExtensor:
    """Integrated magnetic coupling between face currents on a planar thin sheet.

    Each source triangle's Coulomb potential is integrated analytically; three positive
    quadrature points integrate the target triangle. Symmetry enforces magnetic reciprocity,
    and the finite self interaction is integrated exactly. Thickness enters the sheet
    conductivity, not a numerical cutoff in the magnetic kernel.
    """
    coupling = next(magnetic_coupling_blocks(corners, permeability, corners.batch().shape[-1]))
    return (coupling + coupling.adjugate()) / 2


def magnetic_coupling_blocks(corners: Point, permeability: float,
                             block_faces: int) -> Iterator[SparseExtensor]:
    """Target-face blocks of the planar Coulomb integral between the faces of the given corners `[3] Point[F]`,
    before reciprocal symmetrization.

    Summing their pullbacks and then taking the symmetric part gives the same inductance
    as the full face coupling, without keeping all face-pair vector maps in memory.
    """
    edges = corners[[1, 2, 0]] - corners                                   # [edges] Point[F]
    lengths = edges.squared().square_root()                               # [edges] Scalar[F]
    # Each face's area, half the size of the plane two of its edges span, and its sense, the two edges swept
    # through time over twice the area, plus or minus one, intrinsic.
    triangle_areas = edges[0].commutator(edges[1]).norm() / 2             # Scalar[F]
    sense = (edges[1] & edges[2] & time) / (2 * triangle_areas)           # Scalar[F]
    # A triangle's double integral stays finite despite the point kernel's singularity.
    semiperimeter = lengths.sum(axis=0) / 2                                # Scalar[F]
    logarithms = (semiperimeter / (semiperimeter - lengths)).log()
    self_integral = (4 / 3) * triangle_areas.squared() * (logarithms / lengths).sum(axis=0)
    indices = np.arange(corners.batch().shape[-1])
    areas, centres = triangle_areas.batch(), (corners.sum(axis=0) / 3).batch()
    for start in range(0, len(indices), block_faces):
        rows = indices[start:start + block_faces]
        # These positive quadrature points have barycentric weights two-thirds, one-sixth,
        # one-sixth; their equal weights integrate a quadratic exactly.
        targets = (corners.batch()[:, rows] + centres[rows]) / 2           # [samples, block] Point
        # Target points are a batch against the source faces' field: every pair of the two.
        relative = corners - targets[..., None]                           # [samples, block, edges] Point[F]
        distances = relative.squared().square_root()                     # [samples, block, edges] Scalar[F]
        endpoint_sum = distances + distances[..., [1, 2, 0]]               # [samples, block, edges] Scalar[F]
        edge_potential = ((endpoint_sum + lengths) / (endpoint_sum - lengths)).log()
        # Each quadrature point's height above each edge's line, in the face's own sense: the area the
        # corner's offset spans with the edge's unit direction, swept through time.
        heights = (relative & (edges / lengths) & time) / sense          # [samples, block, edges] Scalar[F]
        source_potential = (heights * edge_potential).sum(axis=-1)         # [samples, block] Scalar[F]
        integral = (areas[rows] * source_potential.mean(axis=0)).batch()   # [block, F] Scalar
        integral = integral.at[np.arange(len(rows)), rows].set(self_integral.batch()[rows])
        yield SparseExtensor.from_indices(
            # A current's vector potential: the metric turns the current's direction into a covector,
            # Neumann's pairing of current elements.
            integral * (permeability / (4 * np.pi)) * Current.dual_inverse(),
            rows[:, None], indices[None, :], (len(indices), len(indices)))  # [F, F] Vector <- Current
