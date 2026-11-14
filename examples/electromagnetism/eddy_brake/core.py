"""Eddy-current braking computes the sheet currents, Joule heating, and braking torque in a
conducting disc turning beneath a localized magnetic field.

The disc is modelled in planar spacetime, `x+y+t-`. Displacements across the disc are intrinsic
spatial antivectors (`Point-`), and the magnet's field across the disc is a spatial bivector
(`Bivector+`). A point moves under an angular velocity bivector `spin` by their commutator
`position.commutator(spin)`, and the moving material reads the magnetic field as an electric
spacetime bivector `(velocity & magnetic) ^ t` (`Field+`). Pairing that electric field with an edge
displacement swept along the observer's worldline `time` (`electric & (edge & time)`) reads the
electromotive force along the edge, while each triangle's sheet conductivity maps `Field+` into
sheet current `Current-` (`Antivector-`) and dissipates power `electric & (current & time)`.
Composing conductivity with `(Current & time)` yields the spacetime excitation map into
`Excitation-` (`Antibivector-`), so charge conservation pulls fluxes back to intrinsic vertex
charges (`Amount-`) via `gradient.adjugate()((triangle_areas * excitation_map)(gradient))`, and the
adjugate of `edge_midpoints.commutator(Bivector)` pulls edge Lorentz forces back to the braking
torque (`Moment-`) so `spin & torque == force & velocity`. When self-inductance is included,
resistance and Biot–Savart inductance both act as symmetric forms on closed streamfunctions."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np

from numga import Algebra, NumpyContext, stack
from numga.algebras import STA2D
from numga.sparse import SparseExtensor, spdiag
from numga.topology import Topology

# Planar spacetime of the conducting disc, with spatial directions positive and time negative.
ga = STA2D
Scalar = ga.gatype.scalar()                                                  # Scalar+
Point = ga.gatype.antivector()                                               # Antivector-
Vector = ga.gatype.vector()                                                  # Vector+
Bivector = ga.gatype.bivector()                                              # Bivector+
Rotor = ga.gatype.rotor()                                                    # Rotor+

mv = NumpyContext(ga).multivector
Spatial = Vector.derive.within(Algebra("x+y+"))                              # Vector+
Field = Bivector.derive.touching(Algebra("t-"))                              # Bivector+
Excitation = Field.derive.dual                                               # Antibivector-
Current = Spatial.derive.dual                                                # Antivector-
Amount = ga.gatype.antiscalar()                                              # Antiscalar-
Moment = ga.gatype.antibivector()                                            # Antibivector-
Conductivity = ga.gatype((Current, Field))
Resistivity = ga.gatype((Field, Current))

# Rest-frame worldline direction pairing to +1 with `mv.t` (`mv.t.dual()` carries `-1` from `t-`).
time = -mv.t.dual()                                                          # [] Point-
# Counter-clockwise quarter turn in the disc plane, taking x toward y.
quarter_turn = (mv.xy * (-np.pi / 4)).exp()                                  # [] Rotor+
n_magnetic_block_faces = 64


# --- math -----------------------------------------------------------------------------
def disc_reconstruction(topology: Topology, vertices: Point) -> SparseExtensor:
    """Whitney edge-element reconstruction maps edge voltages `Scalar+[E]` to centroid electric
    fields `Field+[F]` from the simplex gradients of `vertices` (`Point-[V]`)."""
    edge_displacements = topology.d[0] * vertices                            # Point-[E]
    triangle_edges = topology.boundary[2, 1] * edge_displacements            # [3] Point-[F]
    # Joining two oriented boundary edges against `time` gives twice each triangle's signed area.
    double_areas = (triangle_edges[0] & triangle_edges[1]) & time            # Antiscalar-[F]
    # Each corner's simplex gradient is orthogonal to its opposite edge and scales inversely with area.
    gradients = (time & triangle_edges) / double_areas                       # [3] Vector+[F]
    tails, heads = topology.corners[2, 1]                                    # [3], [3]
    spatial = (spdiag((gradients[heads] - gradients[tails]) / 3 * Scalar) * topology.oriented[2, 1]).sum(axis=0)
    return (Vector ^ mv.t)(spatial)                                          # (Field+ <- Scalar+)[F <- E]


def fibre_conductivity(face_centers: Point, principal_conductances: np.ndarray,
                       inner_radius: float, outer_radius: float,
                       mean_conductance: float) -> Conductivity:
    """Radial and circumferential fibre directions across the annulus `[inner_radius, outer_radius]`
    assemble the sheet conductivity `[...] (Current- <- Field+)[F]` from `principal_conductances`
    (`[..., 2]`), with isotropic conductance `mean_conductance` on the inner hub and outer rim."""
    radii = face_centers.antinorm()                                          # Scalar+[F]
    radial = (face_centers / radii).cast(Current)                            # Current-[F]
    tangential = quarter_turn >> radial                                      # Current-[F]
    directions = stack((radial, tangential), axis=-1)                        # [2] Current-[F]
    in_band = (radii >= inner_radius) & (radii <= outer_radius)              # [F]
    fibre_band = mv.scalar(in_band[:, None]).field()                         # Scalar+[F]
    local_conductances = mean_conductance + fibre_band * (principal_conductances - mean_conductance)  # [..., 2] Scalar+[F]
    # Each fibre direction reads the electric field along its worldsheet `directions & time`.
    return (directions * (Field & (directions & time)) * local_conductances).sum(axis=-1)  # [...] (Current- <- Field+)[F]


def steady_response(topology: Topology, edge_displacements: Point, edge_midpoints: Point,
                    triangle_areas: Scalar, reconstruction: SparseExtensor,
                    magnetic: Bivector, spin: Bivector,
                    conductivity: Conductivity) -> tuple[Scalar, Amount, Current, Scalar, Moment]:
    """Charge conservation determines the vertex potential `Scalar+[V]`, dual-edge current flux
    `Amount-[E]`, triangle sheet current `Current-[F]`, Joule heating `Scalar+[F]`, and braking
    torque `Moment-` of a disc turning at `spin` in `magnetic`."""
    V, E, F = topology.n_elements
    # Each edge midpoint moves under an angular velocity bivector by their commutator.
    motion = edge_midpoints.commutator(Bivector)                             # (Point- <- Bivector+)[E]
    velocity = motion(spin)                                                  # [...] Point-[E]
    lorentz = (velocity & magnetic) ^ mv.t                                   # [...] Field+[E]
    emf = lorentz & (edge_displacements & time)                              # [...] Scalar+[E]

    derivative = topology.d[0] * Scalar                                      # Scalar+[E <- V]
    gradient = reconstruction(derivative)                                    # (Field+ <- Scalar+)[F <- V]
    drive = reconstruction(emf)                                              # [...] Field+[F]
    # Sweeping currents along `time` forms the spacetime excitation map into `Excitation-`.
    excitation_map = triangle_areas * (Current & time)(conductivity)         # [...] (Excitation- <- Field+)[F]
    balance = gradient.adjugate()(excitation_map(gradient))                  # [...] (Amount- <- Scalar+)[V <- V]
    # Pinning vertex 0 fixes the free potential constant on the insulated disc without altering edge voltages.
    gauge_vertex = mv.scalar((np.arange(V) == 0)[:, None]).field()           # Scalar+[V]
    gauge = balance.diagonal() * gauge_vertex                                # [...] (Amount- <- Scalar+)[V]
    potential = (balance + gauge).solve(gradient.adjugate()(excitation_map(drive)))  # [...] Scalar+[V]

    voltage = emf - derivative(potential)                                    # [...] Scalar+[E]
    electric = reconstruction(voltage)                                       # [...] Field+[F]
    current = conductivity(electric)                                         # [...] Current-[F]
    flux = reconstruction.adjugate()(triangle_areas * (current & time))      # [...] Amount-[E]

    # Pairing each dual-edge flux with the edge Lorentz 1-form gives the mechanical force covector.
    force = flux & (edge_displacements & magnetic)                           # [...] Spatial+[E]
    torque = motion.adjugate()(force)                                        # [...] Moment-
    heating = triangle_areas * (electric & (current & time))                 # [...] Scalar+[F]
    return potential, flux, current, heating, torque


def closed_currents(topology: Topology, triangle_edges: Point,
                    triangle_areas: Scalar) -> tuple[SparseExtensor, SparseExtensor]:
    """Streamfunctions vanishing on the rim embed from interior vertices via
    `Scalar+[V <- V_interior]` and map to divergence-free sheet currents via
    `(Current- <- Scalar+)[F <- V_interior]`."""
    interior = topology.interior[0]                                          # [V_interior]
    embedder = topology.reading(interior, 0).adjugate() * Scalar             # Scalar+[V <- V_interior]
    rotated_gradients = triangle_edges.cast(Current) / (2 * triangle_areas)  # [3] Current-[F]
    curl = (spdiag(rotated_gradients * Scalar) * topology.incidence[2, 0]).sum(axis=0)  # (Current- <- Scalar+)[F <- V]
    return embedder, curl(embedder)


def streamfunction_projector(topology: Topology, triangle_edges: Point, triangle_areas: Scalar,
                             conductivity: Conductivity) -> Callable[[Current], Scalar]:
    """Resistance-weighted projection maps any sheet current `[...] Current-[F]` to the vertex
    streamfunction `[...] Scalar+[V]` of the nearest closed current vanishing on the rim."""
    embedder, current = closed_currents(topology, triangle_edges, triangle_areas)
    inverse_conductance = (triangle_areas * conductivity).batch().inverse().field()  # [...] (Field+ <- Current-)[F]
    swept = (Current & time)(current)                                        # (Excitation- <- Scalar+)[F <- V_interior]
    pairing = swept.adjugate()(spdiag(triangle_areas ** 2 * inverse_conductance))  # [...] (Amount- <- Current-)[V_interior <- F]
    resistance = pairing(current)                                            # [...] (Amount- <- Scalar+)[V_interior <- V_interior]
    return lambda face_current: embedder(resistance.solve(pairing(face_current)))


def braking(topology: Topology, vertices: Point, edge_displacements: Point, edge_midpoints: Point,
            triangle_edges: Point, triangle_areas: Scalar, reconstruction: SparseExtensor,
            centre: Point, plane: Bivector, field_strength: float, field_width: float,
            conductivity: Conductivity, inertia: float, orientation: Rotor, spin: Bivector,
            heat: Scalar, time_step: float,
            n_steps: int) -> Iterator[tuple[Rotor, Bivector, Point, Scalar, Current, Scalar]]:
    """Yields `(orientation, spin, world_vertices, heat, current, streamfunction)` over `n_steps`
    midpoint braking steps beneath a fixed Gaussian magnetic patch."""
    streamfunctions = streamfunction_projector(topology, triangle_edges, triangle_areas, conductivity)

    def body_response(body_orientation: Rotor,
                      world_spin: Bivector) -> tuple[Scalar, Amount, Current, Scalar, Moment]:
        # Sample the fixed magnet at the rotated edges and pull the field back into the material frame.
        world_edges = body_orientation >> edge_midpoints                     # [...] Point-[E]
        world_field = plane * field_strength * (-(world_edges - centre).antinorm().squared() / (2 * field_width ** 2)).exp()
        magnetic = body_orientation << world_field                           # [...] Bivector+[E]
        body_spin = body_orientation << world_spin                           # [...] Bivector+
        return steady_response(topology, edge_displacements, edge_midpoints, triangle_areas,
                               reconstruction, magnetic, body_spin, conductivity)

    def frame_state(body_orientation: Rotor, world_spin: Bivector,
                    face_heat: Scalar) -> tuple[Rotor, Bivector, Point, Scalar, Current, Scalar]:
        world_vertices = body_orientation >> vertices                        # [...] Point-[V]
        _, _, current, _, _ = body_response(body_orientation, world_spin)    # [...] Current-[F]
        return body_orientation, world_spin, world_vertices, face_heat, current, streamfunctions(current)

    yield frame_state(orientation, spin, heat)
    for _ in range(n_steps):
        # Predict the half-step orientation; linearity in spin gives the drag from one unit-spin solve.
        half_turn = (spin * (-time_step / 4)).exp()                          # [...] Rotor+
        midpoint_orientation = half_turn * orientation                       # [...] Rotor+
        _, _, _, unit_heating, unit_torque = body_response(midpoint_orientation, plane)
        body_plane = midpoint_orientation << plane                           # [...] Bivector+
        drag = -(body_plane & unit_torque)                                   # [...] Scalar+

        # Evaluating torque and heating at the same midpoint spin makes kinetic loss match Joule heat.
        midpoint_spin = spin / (1 + drag * time_step / (2 * inertia))        # [...] Bivector+
        heat = heat + time_step * midpoint_spin.scalar_norm_squared() * unit_heating  # [...] Scalar+[F]
        full_turn = (midpoint_spin * (-time_step / 2)).exp()                 # [...] Rotor+
        orientation = full_turn * orientation                                # [...] Rotor+
        spin = 2 * midpoint_spin - spin                                      # [...] Bivector+
        yield frame_state(orientation, spin, heat)


def magnetic_coupling_blocks(corners: Point, triangle_edges: Point, permeability: float,
                             n_block_faces: int) -> Iterator[SparseExtensor]:
    """Yields `[..., F]` target-face blocks `(Spatial+ <- Current-)[F <- F]` of the planar Biot–Savart
    vector-potential integral from analytic source-triangle potential integrals and three-point target
    quadrature."""
    lengths = triangle_edges.antinorm()                                      # [3] Scalar+[F]
    double_areas = (triangle_edges[0] & triangle_edges[1]) & time            # Antiscalar-[F]
    triangle_areas = double_areas.norm() / 2                                 # Scalar+[F]
    sense = double_areas / (2 * triangle_areas)                              # Antiscalar-[F]
    # Analytic self-integration keeps the 1/r sheet kernel finite on each source triangle.
    semiperimeter = lengths.sum(axis=0) / 2                                  # Scalar+[F]
    logarithms = (semiperimeter / (semiperimeter - lengths)).log()           # [3] Scalar+[F]
    self_integral = (4 / 3) * triangle_areas.squared() * (logarithms / lengths).sum(axis=0)  # Scalar+[F]
    F = corners.batch().shape[-1]
    indices = np.arange(F)                                                   # [F]
    tails, heads = Topology(3, [[0, 1, 2]]).corners[2, 1]                    # [3], [3]
    areas = triangle_areas.batch()                                           # [F] Scalar+
    # Midpoints between corners and centroid form an exact positive quadratic quadrature rule.
    quadrature_targets = (corners.batch() + corners.mean(axis=0).batch()[None, :]) / 2  # [3, F] Point-
    for start in range(0, F, n_block_faces):
        rows = indices[start:start + n_block_faces]
        targets = quadrature_targets[:, rows]                                # [3, ...] Point-
        relative = corners - targets[..., None]                              # [3, ..., 3] Point-[F]
        distances = relative.antinorm()                                      # [3, ..., 3] Scalar+[F]
        endpoint_sum = distances[..., tails] + distances[..., heads]         # [3, ..., 3] Scalar+[F]
        side_potential = ((endpoint_sum + lengths) / (endpoint_sum - lengths)).log()  # [3, ..., 3] Scalar+[F]
        heights = (relative[..., tails] & (triangle_edges / lengths) & time) / sense  # [3, ..., 3] Scalar+[F]
        source_potential = (heights * side_potential).sum(axis=-1)           # [3, ...] Scalar+[F]
        integral = (areas[rows] * source_potential.mean(axis=0)).batch()     # [..., F] Scalar+
        integral = integral.at[np.arange(len(rows)), rows].set(self_integral.batch()[rows])
        yield SparseExtensor.from_indices(
            integral * (permeability / (4 * np.pi)) * Current.dual_inverse(),
            rows[:, None], indices[None, :], (F, F))                         # (Spatial+ <- Current-)[F <- F]


def current_modes(topology: Topology, corners: Point, triangle_edges: Point,
                  triangle_areas: Scalar, conductivity: Conductivity,
                  permeability: float,
                  n_modes: int) -> tuple[Scalar, Scalar, SparseExtensor, SparseExtensor, Resistivity]:
    """Generalized eigenmodes of sheet resistance against Biot–Savart inductance give the `n_modes`
    slowest resistive decay rates `[..., n_modes] Scalar+`, inductance-orthonormal interior
    streamfunctions `[..., n_modes] Scalar+[V_interior]`, the closed-current and vertex-embedding
    maps, and the sheet resistivity `[...] (Field+ <- Current-)[F]`."""
    embedder, current = closed_currents(topology, triangle_edges, triangle_areas)
    resistivity = conductivity.batch().inverse().field()                     # [...] (Field+ <- Current-)[F]
    swept = (Current & time)(current)                                        # (Excitation- <- Scalar+)[F <- V_interior]
    resistance = swept.adjugate()((triangle_areas * resistivity)(current))   # [...] (Amount- <- Scalar+)[V_interior <- V_interior]

    blocks = magnetic_coupling_blocks(corners, triangle_edges, permeability, n_magnetic_block_faces)
    pairing = current.adjugate()                                             # (Amount- <- Spatial+)[V_interior <- F]
    inductance = pairing(next(blocks)(current))                              # (Amount- <- Scalar+)[V_interior <- V_interior]
    for coupling in blocks:
        inductance = inductance + pairing(coupling(current))
    inductance = (inductance + inductance.adjugate()) / 2                    # (Amount- <- Scalar+)[V_interior <- V_interior]
    decay_rates, mode_shapes = resistance.eigh(inductance, n_modes)          # [..., n_modes] Scalar+, [..., n_modes] Scalar+[V_interior]
    return decay_rates, mode_shapes, current, embedder, resistivity


def periodic_drive(face_centers: Point, triangle_areas: Scalar, mode_shapes: Scalar,
                   current_map: SparseExtensor, centre: Point, plane: Bivector,
                   field_strength: float, field_width: float, orientation: Rotor,
                   n_harmonics: int) -> Callable[[Scalar], Scalar]:
    """Fourier projection over one revolution builds a callable mapping rotation angle `[...] Scalar+`
    to modal electromotive forcing `[..., n_modes] Scalar+` up to `n_harmonics` harmonics."""
    n_samples = 2 * n_harmonics + 1
    angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)            # [n_samples]
    turns = (plane * (-angles / 2)).exp()                                    # [n_samples] Rotor+
    orientations = turns[:, None] * orientation                              # [n_samples, ...] Rotor+
    positions = orientations >> face_centers                                 # [n_samples, ...] Point-[F]
    world_field = plane * field_strength * (-(positions - centre).antinorm().squared() / (2 * field_width ** 2)).exp()
    magnetic = orientations << world_field                                   # [n_samples, ...] Bivector+[F]
    body_plane = orientation << plane                                        # [...] Bivector+
    velocity = face_centers.commutator(body_plane)                           # [...] Point-[F]
    electric = (velocity & magnetic) ^ mv.t                                  # [n_samples, ...] Field+[F]
    swept = (Current & time)(current_map)                                    # (Excitation- <- Scalar+)[F <- V_interior]
    loads = swept.adjugate()(triangle_areas * electric)                      # [n_samples, ...] Amount-[V_interior]
    samples = (mode_shapes & loads[..., None]).sites.sum()                   # [n_samples, ..., n_modes] Scalar+

    harmonics = np.arange(1, n_harmonics + 1)                                # [n_harmonics]
    phases = angles[:, None] * harmonics                                     # [n_samples, n_harmonics]
    mean_drive = samples.mean(axis=0)                                        # [..., n_modes] Scalar+
    cosine_drive = 2 * (samples[..., None] * np.cos(phases)[:, None, None]).mean(axis=0)  # [..., n_modes, n_harmonics] Scalar+
    sine_drive = 2 * (samples[..., None] * np.sin(phases)[:, None, None]).mean(axis=0)    # [..., n_modes, n_harmonics] Scalar+

    def evaluate_drive(angle: Scalar) -> Scalar:
        harmonic_phases = angle[..., None, None] * harmonics                 # [..., 1, n_harmonics] Scalar+
        return mean_drive + (harmonic_phases.cos() * cosine_drive + harmonic_phases.sin() * sine_drive).sum(axis=-1)

    return evaluate_drive


def inductive_braking(vertices: Point, triangle_areas: Scalar, decay_rates: Scalar,
                      mode_shapes: Scalar, current_map: SparseExtensor,
                      embedder: SparseExtensor, resistivity: Resistivity,
                      forcing: Callable[[Scalar], Scalar], plane: Bivector, inertia: float,
                      orientation: Rotor, spin: Bivector, heat: Scalar, amplitudes: Scalar,
                      motor_torque: Scalar,
                      time_step: float) -> Iterator[tuple[Rotor, Bivector, Point, Scalar, Current, Scalar, Scalar, Scalar, Scalar]]:
    """Yields `(orientation, spin, world_vertices, heat, current, streamfunction, magnetic_energy,
    thermal_energy, motor_work)` across coupled midpoint substeps driven by `motor_torque`
    (`[n_steps, n_substeps, ...] Scalar+`)."""
    n_substeps = motor_torque.shape[1]
    interval = time_step / n_substeps
    angle = mv.scalar([0]).broadcast_to(orientation.shape)                   # [...] Scalar+
    thermal_energy = heat.sites.sum()                                        # [...] Scalar+
    motor_work = thermal_energy * 0                                          # [...] Scalar+
    factors = (amplitudes * 0).broadcast_to((n_substeps,) + amplitudes.shape)  # [n_substeps, ..., n_modes] Scalar+
    damping = 1 + interval * decay_rates / 2                                 # [..., n_modes] Scalar+

    def modal_state(body_orientation: Rotor, world_spin: Bivector, face_heat: Scalar,
                    modal_amplitudes: Scalar, total_heat: Scalar,
                    work: Scalar) -> tuple[Rotor, Bivector, Point, Scalar, Current, Scalar, Scalar, Scalar, Scalar]:
        world_vertices = body_orientation >> vertices                        # [...] Point-[V]
        magnetic_energy = modal_amplitudes.squared().sum(axis=-1) / 2        # [...] Scalar+
        interior = (modal_amplitudes * mode_shapes).sum(axis=-1)             # [...] Scalar+[V_interior]
        return (body_orientation, world_spin, world_vertices, face_heat, current_map(interior),
                embedder(interior), magnetic_energy, total_heat, work)

    yield modal_state(orientation, spin, heat, amplitudes, thermal_energy, motor_work)
    for step_torques in motor_torque:
        for substep, torque_drive in enumerate(step_torques):
            speed = -(plane | spin)                                          # [...] Scalar+
            drive = forcing(angle + speed * (interval / 2))                  # [..., n_modes] Scalar+
            free_amplitudes = amplitudes / damping                           # [..., n_modes] Scalar+
            driven_amplitudes = interval / 2 * drive / damping               # [..., n_modes] Scalar+
            # Eliminate midpoint modal amplitudes from angular momentum so stored magnetic energy can return to spin.
            midpoint_speed = (speed + interval / (2 * inertia) * (torque_drive - (drive * free_amplitudes).sum(axis=-1))) / (
                1 + interval / (2 * inertia) * (drive * driven_amplitudes).sum(axis=-1))  # [...] Scalar+
            midpoint_amplitudes = free_amplitudes + midpoint_speed[..., None] * driven_amplitudes  # [..., n_modes] Scalar+
            # Retaining substep amplitude factors reconstructs cross-mode face heating without a dense mode covariance.
            factors = factors.at[substep].set(np.sqrt(interval) * midpoint_amplitudes)
            thermal_energy = thermal_energy + interval * (decay_rates * midpoint_amplitudes.squared()).sum(axis=-1)  # [...] Scalar+
            motor_work = motor_work + interval * torque_drive * midpoint_speed  # [...] Scalar+
            amplitudes = 2 * midpoint_amplitudes - amplitudes                # [..., n_modes] Scalar+
            angle = angle + interval * midpoint_speed                        # [...] Scalar+
            midpoint_spin = plane * midpoint_speed                           # [...] Bivector+
            substep_turn = (midpoint_spin * (-interval / 2)).exp()           # [...] Rotor+
            orientation = substep_turn * orientation                         # [...] Rotor+
            spin = 2 * midpoint_spin - spin                                  # [...] Bivector+
        substep_currents = current_map((factors * mode_shapes).sum(axis=-1))  # [n_substeps, ...] Current-[F]
        heat = heat + triangle_areas * (resistivity(substep_currents) & (substep_currents & time)).sum(axis=0)  # [...] Scalar+[F]
        yield modal_state(orientation, spin, heat, amplitudes, thermal_energy, motor_work)
