"""A dynamo: a stirred flow of conducting fluid in a ball that makes a seed magnetic field grow until the field's
force on the current it drives brakes the flow, and the two settle into trading energy back and forth.

Spacetime, `x+y+z+t-`, holds the ball in one instant: its points are directions of space, and the magnetic field is a
plane of space that reads its flux through an oriented patch as `field & patch`. The field is the curl of a potential
on the edges, Whitney's edge forms reconstructing it in each tetrahedron, so it has no sources however it grows. The
moving fluid reads the field as an electric one, the magnetic plane met with the velocity and swept through time,
`(velocity & magnetic) ^ mv.t`, and drives a current with it; the current, gathered back onto the edges, changes the
potential. A conducting core holds the flow, and a mantle of poor conductor around it lets the field reach out as it
would into an insulator. The flow is a few modes of the fluid shell between a solid inner core and the mantle: it
carries itself, the stirring pulls it back toward a helical flow, and the field pushes on the current, the adjugate of
the motional map reading the current as a force.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import Algebra, NumpyContext, stack
from numga.algebras import STA as ga
from numga.sparse import SparseExtensor, spdiag
from examples.util.topology import Topology

mv = NumpyContext(ga).multivector
# Space, within spacetime: the directions x, y and z.
space = Algebra("x+y+z+")
Scalar = ga.gatype.scalar()                         # Scalar+
# The planes of space, which read a point's coordinates; a function's gradient in a tetrahedron is one.
Plane = ga.gatype.vector().derive.within(space)     # Vector+
# The points of space, each the direction from the origin to it; edges, velocities and currents are points too.
Point = Plane.derive.dual                           # Antivector-
# Electric and magnetic fields, planes through time and planes of space.
Field = ga.gatype.bivector()                        # Bivector+
# The magnetic field alone, a plane of space.
Magnetic = Field.derive.within(space)               # Bivector+
# A rotation rate, or the fluid's vorticity: a plane of space.
Rate = Field.derive.within(space)                   # Bivector+
# A function integrated over a share of the ball is an amount.
Amount = ga.gatype.antiscalar()                     # Antiscalar-
# The observer's direction of time.
time = -mv.t.dual()                                 # [] Point-
# The electric field along a direction: the plane it sweeps through time, met with the dual.
along = (Point & time).dual()                       # [] Field+ <- Point-


class Ball:
    """A tetrahedral ball's Whitney operators: the potential's curl, the magnetic energy, and the electric field each
    edge's potential sets up in each tetrahedron."""

    def __init__(self, topology: Topology, vertices: Point):
        self.topology = topology
        self.centres = topology.incidence[3, 0].mean(axis=0) * vertices          # Point-[tets]
        self.edges = topology.d[0] * vertices                                    # Point-[edges]
        # Each tetrahedron's edges in its own vertex order; the first three run from its first corner.
        edges = topology.oriented[3, 1] * self.edges                             # [tet_edges] Point-[tets]
        volumes = (edges[0] & edges[1] & edges[2]) / 6                           # Antitrivector-[tets]
        # The prism each tetrahedron sweeps in a unit of time; its size is the tetrahedron's volume.
        self.prisms = volumes & time                                             # Scalar-[tets]
        sides = topology.boundary[2, 1] * self.edges                             # [3] Point-[faces]
        face_patches = topology.boundary[3, 2] * ((sides[0] & sides[1]) / 2)     # [corners] Patch-[tets]
        # A corner hat's gradient is the volume its opposite face sweeps through time over the prism.
        corner_gradients = (face_patches & time) / (3 * self.prisms)             # [corners] Plane+[tets]
        tails, heads = topology.corners[3, 1]
        # At each centroid an edge's Whitney form is a quarter of the difference of its ends' gradients, and its curl
        # is twice their wedge.
        centroid_forms = (corner_gradients[heads] - corner_gradients[tails]) / 4    # [tet_edges] Plane+[tets]
        self.whitney = (spdiag(centroid_forms * Scalar) * topology.oriented[3, 1]).sum(axis=0)   # (Plane+ <- Scalar+)[tets <- edges]
        curls = 2 * (corner_gradients[tails] ^ corner_gradients[heads])          # [tet_edges] Magnetic+[tets]
        self.curl = (spdiag(curls * Scalar) * topology.oriented[3, 1]).sum(axis=0)   # (Magnetic+ <- Scalar+)[tets <- edges]
        # Each edge's potential changing in time sets up an electric field, its Whitney form swept through time.
        ones = mv.scalar(np.ones((topology.n_elements[3], 1))).field()          # Scalar+[tets]
        sweep = spdiag(ones * (Plane ^ mv.t))                                    # (Field+ <- Plane+)[tets <- tets]
        self.electric = sweep(self.whitney)                                      # (Field+ <- Scalar+)[tets <- edges]
        # Twice the magnetic energy: the field paired with its dual over each volume, pulled back to the edges.
        vacuum = spdiag(self.prisms.norm() * Magnetic.dual())                    # (Patch- <- Magnetic+)[tets <- tets]
        self.stiffness = self.curl.adjugate()(vacuum(self.curl))                 # (Amount- <- Scalar+)[edges <- edges]


def ohm(ball: Ball, conductivity: Scalar) -> SparseExtensor:
    """Ohm's law in each tetrahedron: the current a field drives, the field read back along the direction it points,
    swept through time and weighted by the volume, so that a field paired with it is the heat it makes."""
    return spdiag(ball.prisms.norm() * conductivity * (along.inverse() & time))  # (Patch- <- Field+)[tets <- tets]


def induction(ball: Ball, velocities: Point, conductivity: Scalar) -> tuple[SparseExtensor, SparseExtensor]:
    """The two forms of the potential's induction law, both on the edges: the conductance, which a changing potential
    meets, and the drive, the moving fluid's electric field pushing current along each edge. A potential `A` changes by
    `conductance(dA/dt) == drive(A) - stiffness(A)`."""
    current = ohm(ball, conductivity)                                            # (Patch- <- Field+)[tets <- tets]
    conductance = ball.electric.adjugate()(current(ball.electric))               # (Amount- <- Scalar+)[edges <- edges]
    # The moving fluid reads the magnetic plane as an electric field: met with the velocity, swept through time.
    motional = spdiag((velocities & Magnetic) ^ mv.t)(ball.curl)                 # (Field+ <- Scalar+)[tets <- edges]
    drive = ball.electric.adjugate()(current(motional))                          # (Amount- <- Scalar+)[edges <- edges]
    return conductance, drive


class Flow:
    """A flow in the fluid part of the ball as the amplitudes of its slowest modes, orthonormal in kinetic energy. The
    flow's degrees of freedom are its circulations along the fluid's edges: their Whitney forms are its momentum in each
    tetrahedron, the momentum's dual its velocity, and their curl its vorticity. The modes start from the fluid's
    slowest scalar modes, each laid along each axis as a momentum and made incompressible by taking away the gradient
    that carries its inflow; among the flows these span, the modes are those of least enstrophy for their kinetic
    energy, and the ratio of the two is each mode's rate of viscous decay at unit viscosity."""

    def __init__(self, fluid: Ball, embedding: SparseExtensor, n_scalar_modes: int, n_modes: int):
        topology, volumes = fluid.topology, fluid.prisms.norm()                 # Scalar+[fluid]
        ones = mv.scalar(np.ones((topology.n_elements[3], 1))).field()          # Scalar+[fluid]
        circulating = spdiag(ones * Plane.dual())(fluid.whitney)                 # (Point- <- Scalar+)[fluid <- edges]
        momentum = spdiag(volumes * Point.dual())                                # (Plane+ <- Point-)[fluid <- fluid]
        kinetic = circulating.adjugate()(momentum(circulating))                  # (Amount- <- Scalar+)[edges <- edges]
        # The fluid flowing into each vertex's share of the fluid, and the stiffness of a pressure pushing it down its
        # gradient, with the pressure's free constant pinned at the first vertex.
        inflow = ~topology.d[0] * kinetic                                        # (Amount- <- Scalar+)[vertices <- edges]
        pressure = inflow * topology.d[0]                                        # (Amount- <- Scalar+)[vertices <- vertices]
        reference = mv.scalar((np.arange(topology.n_vertices) == 0)[:, None]).field()   # Scalar+[vertices]
        pinned = pressure + spdiag(pressure.diagonal() * reference)              # (Amount- <- Scalar+)[vertices <- vertices]
        shares = spdiag((~topology.incidence[3, 0].mean(axis=0) * volumes) * Scalar.dual())   # (Amount- <- Scalar+)[vertices <- vertices]
        _, scalar_modes = pressure.eigh(shares, n_scalar_modes + 1)              # [scalar + 1] Scalar+[vertices]
        # Each scalar mode but the constant, at the edges' midpoints, times each edge's reach along each axis.
        means = topology.incidence[1, 0].mean(axis=0) * scalar_modes[1:]         # [scalar] Scalar+[edges]
        reaches = stack((mv.x, mv.y, mv.z))[:, None] & fluid.edges                # [3, 1] Scalar+[edges]
        seeds = (means * reaches).reshape(-1)                                    # [seeds] Scalar+[edges]
        incompressible = seeds - topology.d[0] * pinned.solve(inflow(seeds))     # [seeds] Scalar+[edges]
        # Twice the kinetic energy and twice the enstrophy of the flows the seeds span, as forms on their mixtures.
        spanned = (incompressible * Scalar).batch().field(0, 1).adjoint()        # (Scalar+ <- Scalar+)[edges <- seeds]
        energy = (Scalar & (spanned.adjugate()(kinetic(incompressible)).batch() * Scalar)).field(1, 2)   # Scalar+ <- (Scalar+[seeds], Scalar+[seeds])
        enstrophy = (Scalar & (spanned.adjugate()(fluid.stiffness(incompressible)).batch() * Scalar)).field(1, 2)   # Scalar+ <- (Scalar+[seeds], Scalar+[seeds])
        decay, mixtures = enstrophy.eigh(energy)                                 # [seeds] Scalar+, [seeds] Scalar+[seeds]
        self.decay = decay[:n_modes].field()                                     # Scalar+[modes]
        self.modes = spanned(mixtures[:n_modes])                                 # [modes] Scalar+[edges]
        self.circulation = (self.modes * Scalar).batch().field(0, 1).adjoint()  # (Scalar+ <- Scalar+)[edges <- modes]
        # The velocity and the vorticity of the fluid's circulations, placed among the ball's tetrahedra.
        self.velocity = embedding * circulating                                  # (Point- <- Scalar+)[tets <- edges]
        self.vorticity = embedding * fluid.curl                                  # (Rate+ <- Scalar+)[tets <- edges]

    def reading(self, form: Plane) -> Scalar:
        """Each mode's share of a form on the velocities in the ball's tetrahedra, `Scalar+[modes]`: the power a force
        puts into each mode, or a momentum's amplitude in each."""
        return (self.modes & self.velocity.adjugate()(form)).sites.sum().field()


def advected(ball: Ball, flow: Flow, stirred: Point, amplitudes: Scalar, potential: Scalar) -> tuple[Field, Scalar]:
    """The flow's own part of a step: the motional field of its departure from the stirred flow, `Field+[tets]`, and
    the rate of each mode's amplitude as the flow carries itself, `Scalar+[modes]`. The flow's momentum changes against
    its velocity met with its vorticity plane."""
    circulations = flow.circulation(amplitudes)                                 # Scalar+[edges]
    velocity, vorticity = flow.velocity(circulations), flow.vorticity(circulations)   # Point-[tets], Rate+[tets]
    # The moving fluid reads the magnetic plane as an electric field: met with the velocity, swept through time.
    departure = spdiag((Point & ball.curl(potential)) ^ mv.t)(velocity - stirred)   # Field+[tets]
    return departure, flow.reading(-(ball.prisms.norm() * (velocity & vorticity)))


def dynamo(ball: Ball, flow: Flow, conductivity: Scalar, viscosity: float, pull: float, amplitudes: Scalar,
           potential: Scalar, dt: float, n_steps: int) -> Iterator[tuple[Scalar, Scalar]]:
    """Successive flow amplitudes and potentials, `Scalar+[modes]` and `Scalar+[edges]`, from the stirred flow
    `amplitudes`, which the stirring pulls the flow back toward at the rate `pull`. The field is stepped by
    Crank–Nicolson under the stirred flow, so that every step is a substitution into one factorized map, and by
    Adams–Bashforth under the flow's departure from it. The current of that step brakes the flow
    halfway through it; the flow is stepped by Crank–Nicolson under viscosity and the stirring's pull and by
    Adams–Bashforth as it carries itself."""
    current = ohm(ball, conductivity)                                            # (Patch- <- Field+)[tets <- tets]
    stirred = flow.velocity(flow.circulation(amplitudes))                       # Point-[tets]
    conductance, drive = induction(ball, stirred, conductivity)
    growth = drive - ball.stiffness                                              # (Amount- <- Scalar+)[edges <- edges]
    ahead, behind = conductance - (dt / 2) * growth, conductance + (dt / 2) * growth
    damping = viscosity * flow.decay + pull                                      # Scalar+[modes]
    stirring = pull * amplitudes                                                 # Scalar+[modes]
    departure, rate = advected(ball, flow, stirred, amplitudes, potential)       # Field+[tets], Scalar+[modes]
    previous_departure, previous_rate = departure, rate                          # Field+[tets], Scalar+[modes]
    yield amplitudes, potential
    for _ in range(n_steps):
        pushed = 1.5 * departure - 0.5 * previous_departure                      # Field+[tets]
        stepped = ahead.solve(behind(potential) + dt * ball.electric.adjugate()(current(pushed)))   # Scalar+[edges]
        change, middle = (stepped - potential) / dt, (stepped + potential) / 2   # Scalar+[edges], Scalar+[edges]
        # The current of the step, the one Ampère's law balances, `electric.adjugate()(currents) == stiffness(middle)`:
        # the motional field less the field the changing potential sets up.
        motional = spdiag((Point & ball.curl(middle)) ^ mv.t)                   # (Field+ <- Point-)[tets <- tets]
        currents = current(motional(stirred) + pushed - ball.electric(change))  # Patch-[tets]
        # The field's force on the current is the adjugate of the motional map, which reads the current as a form on
        # velocities.
        braking = flow.reading(-motional.adjugate()(currents))                   # Scalar+[modes]
        pushed_flow = amplitudes + dt * (1.5 * rate - 0.5 * previous_rate + braking + stirring - damping * amplitudes / 2)   # Scalar+[modes]
        amplitudes, potential = pushed_flow / (1 + (dt / 2) * damping), stepped  # Scalar+[modes], Scalar+[edges]
        previous_departure, previous_rate = departure, rate
        departure, rate = advected(ball, flow, stirred, amplitudes, potential)
        yield amplitudes, potential
