"""Hydromagnetic dynamo in a planetary core in spacetime.

Stirred conducting fluid in an outer-core shell stretches and twists a weak seed magnetic field until the magnetic force on the induced conduction current brakes the flow and the two settle into exchanging energy. Spatial planes, displacements, velocities, and conduction 4-currents live in spacetime, where edge potentials reconstruct divergence-free magnetic planes in each tetrahedron through Whitney forms built from corner simplex gradients. Moving fluid meets the magnetic plane in a motional electric field that drives conduction 4-currents, and pairing the magnetic plane back with those 4-currents gives the Lorentz braking form on fluid velocities.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import Algebra, NumpyContext, stack
from numga.algebras import STA as ga
from numga.sparse import Sparse, SparseExtensor, spdiag
from numga.topology import Topology

mv = NumpyContext(ga).multivector
# Spatial planes within spacetime span x, y, and z.
space = Algebra("x+y+z+")
Scalar = ga.gatype.scalar()                         # Scalar+
Plane = ga.gatype.vector().derive.within(space)     # Vector+
Point = Plane.derive.dual                           # Antivector-
Field = ga.gatype.bivector()                        # Bivector+
Magnetic = Field.derive.within(space)               # Bivector+
Rate = Field.derive.within(space)                   # Bivector+
# Rest-frame observer 4-velocity and rest-frame conduction 4-current readout.
time = -mv.t.dual()                                 # Antivector-
electric_current = -(Field & time).dual()           # Point- <- Field+


def whitney_operators(topology: Topology, vertices: Point) -> tuple[Scalar, SparseExtensor, SparseExtensor]:
    """Tetrahedral volumes (`Scalar+[T]`), centroid Whitney 1-forms (`(Plane+ <- Scalar+)[T <- E]`), and curls (`(Magnetic+ <- Scalar+)[T <- E]`) map edge circulations onto each tetrahedron."""
    edges = topology.d[0] * vertices                                                    # Point-[E]
    simplex_edges = topology.oriented[3, 1] * edges                                     # [6] Point-[T]
    prisms = ((simplex_edges[0] & simplex_edges[1] & simplex_edges[2]) / 6) & time      # Scalar-[T]
    sides = topology.boundary[2, 1] * edges                                             # [3] Point-[F]
    face_patches = topology.boundary[3, 2] * ((sides[0] & sides[1]) / 2)                # [4] Antibivector-[T]
    # Each corner's simplex gradient is the spacetime volume its opposite face sweeps over three prism volumes.
    gradients = (face_patches & time) / (3 * prisms)                                    # [4] Plane+[T]
    tails, heads = topology.corners[3, 1]                                               # [6], [6]
    # At each centroid an edge's Whitney form is a quarter of its endpoint gradient difference, and its curl is twice their wedge.
    centroid_forms = (gradients[heads] - gradients[tails]) / 4                          # [6] Plane+[T]
    whitney = (spdiag(centroid_forms * Scalar) * topology.oriented[3, 1]).sum(axis=0)   # (Plane+ <- Scalar+)[T <- E]
    curls = 2 * (gradients[tails] ^ gradients[heads])                                   # [6] Magnetic+[T]
    curl = (spdiag(curls * Scalar) * topology.oriented[3, 1]).sum(axis=0)               # (Magnetic+ <- Scalar+)[T <- E]
    return prisms.norm(), whitney, curl


def magnetic_stiffness(volumes: Scalar, curl: SparseExtensor) -> SparseExtensor:
    """Pulling the vacuum excitation back across the curl maps edge potentials to edge flux forms (`(Antiscalar- <- Scalar+)[E <- E]`)."""
    excitation = volumes * Magnetic.dual()                                              # (Antibivector- <- Magnetic+)[T]
    return curl.adjugate()(excitation(curl))                                            # (Antiscalar- <- Scalar+)[E <- E]


def flow_modes(fluid_topology: Topology, fluid_vertices: Point, embedding: Sparse,
               n_scalar_modes: int, n_modes: int) -> tuple[Scalar, Point, Rate]:
    """Incompressible eigenmodes of the fluid shell give viscous decay rates (`[n_modes] Scalar+`), velocities (`[n_modes] Point-[T]`), and vorticities (`[n_modes] Rate+[T]`)."""
    V_fluid, E_fluid, F_fluid, T_fluid = fluid_topology.n_elements
    volumes, whitney, curl = whitney_operators(fluid_topology, fluid_vertices)          # Scalar+[T_fluid], (Plane+ <- Scalar+)[T_fluid <- E_fluid], (Magnetic+ <- Scalar+)[T_fluid <- E_fluid]
    velocity_whitney = Plane.dual()(whitney)                                            # (Point- <- Scalar+)[T_fluid <- E_fluid]
    momentum = volumes * Point.dual()                                                   # (Plane+ <- Point-)[T_fluid]
    kinetic = velocity_whitney.adjugate()(momentum(velocity_whitney))                   # (Antiscalar- <- Scalar+)[E_fluid <- E_fluid]
    # Divergence at each vertex and the pressure stiffness that removes inflow, pinned at vertex zero.
    inflow = fluid_topology.d[0].adjugate() * kinetic                                   # (Antiscalar- <- Scalar+)[V_fluid <- E_fluid]
    pressure = inflow * fluid_topology.d[0]                                             # (Antiscalar- <- Scalar+)[V_fluid <- V_fluid]
    reference = mv.scalar((np.arange(V_fluid) == 0)[:, None]).field()                   # Scalar+[V_fluid]
    pinned_pressure = pressure + pressure.diagonal() * reference                        # (Antiscalar- <- Scalar+)[V_fluid <- V_fluid]
    vertex_volumes = spdiag((fluid_topology.incidence[3, 0].mean(axis=0).adjugate() * volumes) * Scalar.dual())  # (Antiscalar- <- Scalar+)[V_fluid <- V_fluid]
    _, scalar_modes = pressure.eigh(vertex_volumes, n_scalar_modes + 1)                 # [n_scalar_modes + 1] Scalar+[V_fluid]
    # Non-constant scalar modes at edge midpoints laid along x, y, and z and projected free of inflow.
    edge_means = fluid_topology.incidence[1, 0].mean(axis=0) * scalar_modes[1:]         # [n_scalar_modes] Scalar+[E_fluid]
    edge_spans = stack((mv.x, mv.y, mv.z))[:, None] & (fluid_topology.d[0] * fluid_vertices)  # [3, 1] Scalar+[E_fluid]
    seed_circulations = (edge_means * edge_spans).reshape(-1)                           # [3 * n_scalar_modes] Scalar+[E_fluid]
    solenoidal_flows = seed_circulations - fluid_topology.d[0] * pinned_pressure.solve(inflow(seed_circulations))  # [3 * n_scalar_modes] Scalar+[E_fluid]
    # Generalized eigenmodes of enstrophy against kinetic energy on the solenoidal subspace.
    basis_map = (solenoidal_flows * Scalar).batch().field(0, 1).adjoint()               # (Scalar+ <- Scalar+)[E_fluid <- 3 * n_scalar_modes]
    stiffness = magnetic_stiffness(volumes, curl)                                       # (Antiscalar- <- Scalar+)[E_fluid <- E_fluid]
    energy_form = (Scalar & (basis_map.adjugate()(kinetic(solenoidal_flows)).batch() * Scalar)).field(1, 2)     # (Scalar+ <- (Scalar+, Scalar+))[3 * n_scalar_modes, 3 * n_scalar_modes]
    enstrophy_form = (Scalar & (basis_map.adjugate()(stiffness(solenoidal_flows)).batch() * Scalar)).field(1, 2)  # (Scalar+ <- (Scalar+, Scalar+))[3 * n_scalar_modes, 3 * n_scalar_modes]
    decay_rates, mixtures = enstrophy_form.eigh(energy_form)                            # [3 * n_scalar_modes] Scalar+, [3 * n_scalar_modes] Scalar+[3 * n_scalar_modes]
    circulations = basis_map(mixtures[:n_modes])                                        # [n_modes] Scalar+[E_fluid]
    return decay_rates[:n_modes], (embedding * velocity_whitney)(circulations), (embedding * curl)(circulations)


def induction_operators(volumes: Scalar, whitney: SparseExtensor, curl: SparseExtensor,
                        velocities: Point, conductivity: Scalar) -> tuple[SparseExtensor, SparseExtensor]:
    """Edge conductance and motional drive (`(Antiscalar- <- Scalar+)[E <- E]`) pull Ohm's conduction 4-current back onto the edges."""
    conduction = (volumes * conductivity) * electric_current                            # (Point- <- Field+)[T]
    electric = (Plane ^ mv.t)(whitney)                                                  # (Field+ <- Scalar+)[T <- E]
    conductance = whitney.adjugate()(conduction(electric))                              # (Antiscalar- <- Scalar+)[E <- E]
    # Moving fluid reads the magnetic plane as an electric field by meeting with velocity and sweeping through time.
    motional = ((velocities & Magnetic) ^ mv.t)(curl)                                   # (Field+ <- Scalar+)[T <- E]
    drive = whitney.adjugate()(conduction(motional))                                    # (Antiscalar- <- Scalar+)[E <- E]
    return conductance, drive


def flow_advection(volumes: Scalar, curl: SparseExtensor, mode_velocities: Point, mode_vorticities: Rate,
                   target_velocity: Point, amplitudes: Scalar, potential: Scalar) -> tuple[Field, Scalar]:
    """Departure motional field (`Field+[T]`) and self-advection mode rates (`[n_modes] Scalar+`) come from the current flow state."""
    velocity = (amplitudes * mode_velocities).sum(axis=0)                               # Point-[T]
    vorticity = (amplitudes * mode_vorticities).sum(axis=0)                             # Rate+[T]
    departure_field = ((velocity - target_velocity) & curl(potential)) ^ mv.t           # Field+[T]
    advection_rates = (-(volumes * (velocity & vorticity)) & mode_velocities).sites.sum()  # [n_modes] Scalar+
    return departure_field, advection_rates


def dynamo_states(volumes: Scalar, whitney: SparseExtensor, curl: SparseExtensor,
                  decay_rates: Scalar, mode_velocities: Point, mode_vorticities: Rate,
                  conductivity: Scalar, viscosity: float, pull: float, target_amplitudes: Scalar,
                  potential: Scalar, dt: float, n_steps: int) -> Iterator[tuple[Scalar, Scalar]]:
    """Yields flow mode amplitudes (`[n_modes] Scalar+`) and edge potentials (`Scalar+[E]`) across coupled Crank–Nicolson and Adams–Bashforth steps."""
    conduction = (volumes * conductivity) * electric_current                            # (Point- <- Field+)[T]
    electric = (Plane ^ mv.t)(whitney)                                                  # (Field+ <- Scalar+)[T <- E]
    target_velocity = (target_amplitudes * mode_velocities).sum(axis=0)                 # Point-[T]
    conductance, drive = induction_operators(volumes, whitney, curl, target_velocity, conductivity)
    growth = drive - magnetic_stiffness(volumes, curl)                                  # (Antiscalar- <- Scalar+)[E <- E]
    implicit_operator = conductance - (dt / 2) * growth                                 # (Antiscalar- <- Scalar+)[E <- E]
    explicit_operator = conductance + (dt / 2) * growth                                 # (Antiscalar- <- Scalar+)[E <- E]
    damping = viscosity * decay_rates + pull                                            # [n_modes] Scalar+
    stirring_rates = pull * target_amplitudes                                           # [n_modes] Scalar+
    amplitudes = target_amplitudes                                                      # [n_modes] Scalar+
    departure_field, advection_rates = flow_advection(
        volumes, curl, mode_velocities, mode_vorticities, target_velocity, amplitudes, potential
    )
    previous_departure, previous_rates = departure_field, advection_rates               # Field+[T], [n_modes] Scalar+
    yield amplitudes, potential
    for _ in range(n_steps):
        extrapolated_field = 1.5 * departure_field - 0.5 * previous_departure           # Field+[T]
        next_potential = implicit_operator.solve(
            explicit_operator(potential) + dt * whitney.adjugate()(conduction(extrapolated_field))
        )                                                                               # Scalar+[E]
        potential_rate = (next_potential - potential) / dt                              # Scalar+[E]
        middle_field = curl((next_potential + potential) / 2)                           # Magnetic+[T]
        # Midpoint conduction 4-current balances Ampère's law between motional and inductive electric fields.
        currents = conduction(((target_velocity & middle_field) ^ mv.t) + extrapolated_field - electric(potential_rate))  # Point-[T]
        # Pairing the magnetic plane with the 4-current yields the Lorentz force 1-form on fluid velocities.
        braking_rates = ((middle_field & currents) & mode_velocities).sites.sum()       # [n_modes] Scalar+
        numerator_amplitudes = amplitudes + dt * (
            1.5 * advection_rates - 0.5 * previous_rates + braking_rates + stirring_rates - damping * amplitudes / 2
        )                                                                               # [n_modes] Scalar+
        amplitudes = numerator_amplitudes / (1 + (dt / 2) * damping)                    # [n_modes] Scalar+
        potential = next_potential                                                      # Scalar+[E]
        previous_departure, previous_rates = departure_field, advection_rates
        departure_field, advection_rates = flow_advection(
            volumes, curl, mode_velocities, mode_vorticities, target_velocity, amplitudes, potential
        )
        yield amplitudes, potential
