"""The Meissner effect: a superconducting rod cooled through its transition in a magnetic field across it expels the
field, where a perfect conductor would freeze it in; on a mesh of triangles over the rod's cross-section, in spacetime.

A superconductor's electrons flow as two fluids: a superfluid, whose density grows from nothing at the transition
temperature as the metal cools, and the normal electrons, which conduct with resistance. The superfluid's momentum
together with its charge times the vector potential, its canonical momentum, is the spacetime gradient of one phase,
so its exterior derivative vanishes, `gradient ^ (momentum + charge * potential) == 0`. The derivative's part through
time and space is London's first equation, the supercurrent accelerated by the electric field, which alone makes a
perfect conductor; its part in space is London's second equation, which fixes the supercurrent by the potential itself
and so expels the field, the Meissner effect.

The algebra is spacetime, x+y+z+t-, with everything independent of z: the potential lies along z, `potential * mv.z`,
so that its level lines in the plane are the magnetic field's lines, the field `gradient ^ (potential * mv.z)` a
bivector in planes through z, and the currents run along z. A phase that does not change along z has no part along z,
so there the canonical momentum's part along z is nothing and the supercurrent is minus the superfluid's charge squared
over its mass times the potential. Ampère's law, the potential's stiffness balancing the current, and the normal
electrons' Ohmic current under the electric field, minus the potential's rate of change, make each step one solve.

Around a superconducting ring the same closure integrates to a whole number: the canonical momentum's circulation is
the phase's winding, so the ring's fluxoid, the flux through it with its current's share, is a whole number of flux
quanta. With the field along the ring's axis, `strength * mv.xy`, its strength screens into the ring by London's second
equation, stands uniform in the hole, and its level lines are the current's lines; the fluxoid fixes the hole's field.

Lengths are in the rod's radii, times in the time a field takes to diffuse across the normal metal, and fields in the
applied field's units, or for the ring in flux quanta per squared ring radius; the superfluid's charge squared over its
mass is written through the London penetration depth.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import Algebra, NumpyContext, stack
from numga.algebras import STA as ga
from numga.sparse import SparseExtensor, spdiag

from examples.util.topology import Topology

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                   # Scalar+
# The planes of space, which read a point's coordinates; a function's gradient on a triangle is one.
Plane = ga.gatype.vector().derive.within(Algebra("x+y+z+"))                   # Vector+
# The points of space, each the direction from the origin to it.
Point = Plane.derive.dual                                                     # Antivector-
# The magnetic field across the rod, in the planes through z.
Field = ga.gatype.bivector().derive.touching(Algebra("x+y+"), Algebra("z+"))  # Bivector+
# The rod's direction and time's, along which nothing changes.
rod_direction = mv.z.dual()                                                   # [] Point-
time = -mv.t.dual()                                                           # [] Antivector-
# A current: a direction of flow swept through time.
Current = ga.gatype.antivector()                                              # Antivector-
# The superfluid's inertia, per squared penetration depth: a current read along each direction of space, in the metal's
# rest frame, into the momentum that carries it.
inertia = (stack((mv.x, mv.y, mv.z)) * (Current & stack((mv.x, mv.y, mv.z)))).sum(axis=0)   # [] Plane+ <- Current-


# --- plumbing -------------------------------------------------------------------------
def assembled(topology: Topology, local: Scalar) -> SparseExtensor:
    """A form on each triangle's corners `[corners, corners] Scalar+[faces]` gathered onto the vertices it joins,
    `Scalar+[vertices <- vertices]`."""
    corners = topology.incidence[2, 0]                                        # [corners] Scalar+[faces <- vertices]
    return (~corners[:, None] * spdiag(local) * corners[None, :]).sum(axis=-1).sum(axis=-1)


# --- math -----------------------------------------------------------------------------
def triangles(topology: Topology, vertices: Point) -> tuple[Scalar, Plane, Scalar]:
    """Each triangle's area `Scalar+[faces]`, each corner's hat gradient `[corners] Plane+[faces]`, the plane through
    the facing side, the rod's direction and time over twice the area, and each vertex's share of the area around it
    `Scalar+[vertices]`."""
    sides = topology.boundary[2, 1] * (topology.d[0] * vertices)[..., None]  # [corners] Point-[faces]
    # Twice a triangle's area: two of its sides joined, swept along the rod and through time.
    double_areas = ((sides[..., 0] & sides[..., 1]) & rod_direction) & time   # Scalar-[faces]
    areas = double_areas.norm() / 2                                           # Scalar+[faces]
    # A corner's hat gradient: the plane through the facing side, along the rod and through time, over twice the area,
    # so that it reads one from the side to the corner.
    gradients = (rod_direction & time & sides) / double_areas[..., None]      # [corners] Plane+[faces]
    return areas, gradients, (~topology.incidence[2, 0] * (areas / 3)).sum(axis=-1)


def cooling(topology: Topology, areas: Scalar, gradients: Plane, cells: Scalar, inside: Scalar, outer: np.ndarray,
            heights: Scalar, penetration: float, superfluid: np.ndarray, applied: np.ndarray, reference: Scalar,
            dt: float, stride: int) -> Iterator[tuple[Scalar, Scalar]]:
    """The potential and the supercurrent `[...] Scalar+[vertices]` every `stride` steps of `dt`, of a rod `inside`
    `Scalar+[vertices]` whose superfluid's share of its electrons is `superfluid` at each step, under an applied field
    of these sizes `[steps]`, held on the outer border `outer` by the potential `applied * heights`; its superfluid's
    momentum responds to the potential's departure from `reference` `[...] Scalar+[vertices]`: nothing for a
    superconductor, whose canonical momentum has no part along z, and the potential when its superfluid formed for a
    perfect conductor, whose momentum only follows the electric field."""
    free = topology.reading(np.setdiff1d(np.arange(topology.n_vertices), outer), 0)   # Scalar+[free <- vertices]
    # The magnetic field of a potential rising by one at each corner, through the plane its gradient sweeps along z, and
    # the potential's stiffness, Ampère's law: each field read against the vacuum's excitation of another.
    fields = gradients ^ mv.z                                                 # [..., corners] Field+[faces]
    stiffness = assembled(topology, (fields[..., :, None] & Field.dual()(fields[..., None, :])) * areas[..., None, None])
    # The normal electrons' conduction over a step, in the rod's cells.
    conducting = spdiag(cells * inside / dt)                                  # Scalar+[vertices <- vertices]
    potential = applied[0] * heights + 0 * reference                          # [...] Scalar+[vertices]
    for step, (share, field) in enumerate(zip(superfluid, applied)):
        # The superfluid's charge squared over its mass, its share of the electrons over the squared penetration depth.
        londons = spdiag(cells * inside * share / penetration ** 2)           # Scalar+[vertices <- vertices]
        held = field * heights                                                # Scalar+[vertices]
        solving = (free * (conducting + stiffness + londons) * ~free) * Scalar
        right = conducting * potential + londons * reference - (conducting + stiffness + londons) * held
        potential = held + ~free * solving.solve(free * right)                # [...] Scalar+[vertices]
        if step % stride == 0:
            yield potential, -(inside * share / penetration ** 2) * (potential - reference)


def ring_fields(topology: Topology, areas: Scalar, gradients: Plane, in_ring: np.ndarray, hole_faces: np.ndarray,
                in_hole: np.ndarray, rim: np.ndarray, penetration: float, quanta: np.ndarray, applied: np.ndarray
                ) -> Scalar:
    """The field's strength along the axis `[...] Scalar+[vertices]` of a ring of triangles `in_ring` `[faces]` around
    a hole of triangles `hole_faces` `[faces]` whose vertices, its edge's included, are `in_hole` `[vertices]`, its rim
    `rim` held at the applied field `[...]`, holding these fluxoids `[...]` in flux quanta: the field that screens into
    the ring by London's second equation from the hole's uniform field and the applied one, with the hole's field
    whose flux, and the circulation of the current along the hole's edge, add up to the fluxoid."""
    corners = topology.incidence[2, 0]                                        # [corners] Scalar+[faces <- vertices]
    ring = mv.scalar(in_ring[:, None] * 1.0).field()                          # Scalar+[faces]
    # The supercurrent a field's strength rising by one at each corner carries by Ampère's law, the hat's gradient
    # wedged with the excitation of a unit field along the axis.
    currents = gradients ^ mv.xy.dual()                                       # [corners] Current-[faces]
    # London's screening on the ring alone: the supercurrent's kinetic energy, each current read against the momentum
    # another carries, times the squared penetration depth, with the field's own energy in the cells.
    local = (currents[..., :, None] & inertia(currents[..., None, :])) * (ring * areas)[..., None, None]
    screening = penetration ** 2 * (~corners[:, None] * spdiag(local) * corners[None, :]).sum(axis=-1).sum(axis=-1) + (
        (~corners * (ring * areas / 3)).sum(axis=-1))                         # Scalar+[vertices <- vertices]
    on_rim = np.isin(np.arange(topology.n_vertices), rim)                     # [vertices]
    free = topology.reading(np.flatnonzero(~(in_hole | on_rim)), 0)           # Scalar+[free <- vertices]
    hole, edge = (mv.scalar(chosen[:, None] * 1.0).field() for chosen in (in_hole, on_rim))   # Scalar+[vertices] each
    # The fields of a unit field in the hole with none at the rim, and of a unit field at the rim with none in the hole.
    units = stack([hole, edge])                                               # [2] Scalar+[vertices]
    screened = units + ~free * (free * screening * ~free * Scalar).solve(-(free * (screening * units)))
    # Each one's fluxoid: its flux through the hole, and its current's circulation along the hole's edge, which is the
    # screening's reaction there.
    hole_area = (mv.scalar(hole_faces[:, None] * 1.0).field() * areas).sites.sum()   # [] Scalar+
    fluxoids = (hole * (screening * screened)).sites.sum() + hole_area * np.array([1.0, 0.0])   # [2] Scalar+
    hole_field = (quanta - applied * fluxoids[1]) / fluxoids[0]               # [...] Scalar+
    return hole_field * screened[0] + applied * screened[1]
