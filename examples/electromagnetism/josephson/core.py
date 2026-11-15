"""Fluxons travel through a planar Josephson junction as relativistic strings of magnetic flux whose local condensate phase difference is a unit direction in an internal plane joined to the junction's spacetime.

Two superconductors separated by a thin barrier couple their condensate phases through Josephson's relations, driving a tunnelling supercurrent proportional to the sine of the phase difference while turning the phase at a rate set by the barrier voltage. Combining those relations with Maxwell's equations in a wide junction yields the two-dimensional sine-Gordon wave equation with characteristic speed equal to the Swihart velocity, where each full turn of the phase carries one magnetic flux quantum per unit length.

Joining the junction's three-dimensional spacetime with an orthogonal Euclidean phase plane models each local phase difference as a unit vector in the phase plane and its angular rate as a phase-plane bivector. Tunnelling pulls each phase direction toward the bias axis while neighbouring directions pull through the triangle simplex gradients paired across the supercurrent duality map, so wedging the phase direction with the net pull produces both torques without unwrapping angles. Because the internal phase plane commutes with the junction's spacetime, spacetime boosts and spatial rotations act directly on junction events while carrying phase directions unchanged.

Lengths are measured in Josephson penetration depths and times in inverse plasma frequencies, setting the Swihart velocity to unity.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import Algebra, NumpyContext
from numga.sparse import SparseExtensor, spdiag
from numga.topology import Topology

# Spacetime of the planar junction joined with an orthogonal Euclidean plane for the condensate phase.
phase_plane = Algebra("a+b+")
ga = Algebra("x+y+t-") * phase_plane
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                   # Scalar+
Plane = ga.gatype.vector()                                                    # Vector+
Point = ga.gatype.antivector()                                                # Antivector-
Phase = Plane.derive.within(phase_plane)                                      # Vector+
Rate = ga.gatype.bivector().derive.within(phase_plane)                        # Bivector+
# Phase-plane rotor turning the zero-phase axis into a local phase direction.
Turn = (mv.ab * 0).exp().gatype                                               # Rotor+
# Unit spacetime displacements along the junction plane and along time.
x_step = mv.x.dual()                                                          # [] Point-
y_step = mv.y.dual()                                                          # [] Point-
time_step = -mv.t.dual()                                                      # [] Point-
# Duality map reading a spatial phase gradient as the supercurrent it drives along the junction.
supercurrent = Plane.dual()                                                   # [] Point- <- Plane+


def simplex_gradients(topology: Topology, vertices: Point) -> tuple[Scalar, Plane]:
    """Boundary edges and oriented triangle patches give the triangle areas (`Scalar+[F]`) and corner simplex gradients (`[3] Plane+[F]`)."""
    sides = topology.boundary[2, 1] * (topology.d[0] * vertices)              # [3] Point-[F]
    patches = (sides[0] & sides[1]) / 2                                       # Antibivector-[F]
    areas = patches.antinorm()                                                # Scalar+[F]
    gradients = (sides | patches.inverse()) / 2                               # [3] Plane+[F]
    return areas, gradients


def tension_operator(topology: Topology, areas: Scalar, gradients: Plane) -> SparseExtensor:
    """Pulling the triangle gradient-supercurrent pairing back to vertices and dividing by lumped dual areas forms the field tension (`Scalar+[V <- V]`)."""
    corners = topology.incidence[2, 0]                                        # [3] Scalar+[F <- V]
    dual_areas = (corners.adjugate() * (areas / 3)).sum(axis=0)               # Scalar+[V]
    # Pair each corner's simplex gradient with the supercurrent driven by another corner's gradient over the triangle.
    local = (gradients[:, None] & supercurrent(gradients[None, :])) * areas   # [3, 3] Scalar+[F]
    stiffness = (corners[:, None].adjugate() * spdiag(local) * corners[None, :]).sum(axis=0).sum(axis=0)
    return spdiag(1 / dual_areas) * stiffness


def fluxon_turn(events: Point, rapidity: np.ndarray, sense: float) -> Turn:
    """Boosting junction events (`Point-[V]`) into the kink rest frame produces the phase-plane rotors (`[...] Turn+[V]`) of a straight fluxon or antifluxon."""
    # Boost spacetime events into the moving kink's rest frame and read their coordinate normal to the kink line.
    boost = (mv.xt * (-rapidity / 2)).exp()                                   # [...] Rotor+
    rest_offsets = mv.x & (boost >> events)                                   # [...] Scalar+[V]
    angles = 4 * (sense * rest_offsets).exp().arctan()                        # [...] Scalar+[V]
    return (mv.ab * (-angles / 2)).exp()


def phase_frames(tension: SparseExtensor, phases: Phase, rates: Rate, dt: float, n_steps: int,
                 stride: int) -> Iterator[Phase]:
    """Yields the vertex phase directions (`[...] Phase+[V]`) every `stride` leapfrog steps of `dt` under Josephson tunnelling and field tension."""
    for step_index in range(n_steps + 1):
        if step_index % stride == 0:
            yield phases
        pull = mv.a - tension * phases                                        # [...] Phase+[V]
        rates = rates + dt * (phases ^ pull)                                  # [...] Rate+[V]
        step_turns = (rates * (-dt / 2)).exp()                                # [...] Turn+[V]
        phases = step_turns >> phases                                         # [...] Phase+[V]


def field_strengths(topology: Topology, gradients: Plane, phases: Phase) -> Scalar:
    """Wedging triangle simplex gradients with vertex phase directions gives the squared magnetic field strength (`[...] Scalar+[F]`) in each triangle."""
    corners = topology.incidence[2, 0]                                        # [3] Scalar+[F <- V]
    corner_phases = corners * phases[..., None]                               # [..., 3] Phase+[F]
    fields = (gradients ^ corner_phases).sum(axis=-1)                         # [...] Bivector+[F]
    return fields.scalar_norm_squared()
