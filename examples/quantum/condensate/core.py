"""A rotating Bose-Einstein condensate in a harmonic trap, settling into its ground state, a lattice of quantized
vortices, on a mesh of triangles; in the plane of the trap joined with a plane of the condensate's own.

A condensate is one quantum state shared by all its atoms, a wave over the trap. It cannot rotate as a body does: its
flow is set by how the wave turns from place to place, so it circulates only around vortices, holes in its density
around which the wave turns once, each holding one quantum of circulation. Set rotating, it holds as many vortices as
the rotation asks, spread so evenly that on average the condensate turns like a rigid body: twice the rotation rate
over the circulation quantum of them per unit area, Feynman's rule.

The algebra is the plane of the trap, x+y+, joined with a plane of the condensate's own, a+b+: `Algebra("x+y+") *
Algebra("a+b+")`. The condensate at each point of the trap is modelled as a vector in its own plane, whose length is the
square root of the density and whose direction is its phase, `amplitude * (((mv.ab) * (-phase / 2)).exp() >> mv.a)`. Two
turnings meet in the rotating trap, each a bivector of its own plane: the trap's, `mv.xy`, which drags each point a
quarter turn about the middle, `point.commutator(mv.xy)`, and the wave's, `mv.ab`, which turns the condensate in its own
plane, `wave * mv.ab`, and commutes with everything in the trap's plane. The energy's rotating part pairs them: the
wave's change along the trap's drag, the drag paired with the wave's gradient in the mixed planes `xa xb ya yb`, against
the wave turned a quarter in its own plane. The ground state is found by descending the energy at a fixed number of
atoms: the kinetic energy, the trap and the rotation in one implicit step, the atoms' repulsion explicit, the wave
rescaled to its atom number after each step. A vortex is a triangle around which the wave turns once: the turns along
its three edges, each the logarithm of the rotor between the wave's directions at the edge's ends, add up to a whole
turn.

Lengths are in the trap's oscillator length, energies in its quantum, and the atom number is one, so that the
circulation quantum is two pi.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import Algebra, NumpyContext
from numga.sparse import SparseExtensor, spdiag

from examples.util.topology import Topology

# The trap's plane, and the condensate's own plane.
ga = Algebra("x+y+") * Algebra("a+b+")
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                   # Scalar+
Plane = ga.gatype.vector()                                                    # Vector+
Point = ga.gatype.antivector()                                                # Antivector-
Wave = ga.gatype.from_blades("a b")                                           # Vector+
Turn = ga.gatype.from_blades("ab")                                            # Bivector+
# Unit steps across and up the trap's plane, as points from its middle.
axis = mv.x.dual()                                                            # [] Point-
across = mv.y.dual()                                                          # [] Point-
# The condensate's own plane, the join of its two directions, which no step across the trap has a part along.
wave_plane = mv.a.dual() & mv.b.dual()                                        # [] Antibivector-
# The condensate's mobility, its inverse mass, one in these units: it turns a gradient of the wave into the flow it
# drives.
mobility = Plane.dual()                                                       # [] Point- <- Plane+


# --- plumbing -------------------------------------------------------------------------
def assembled(topology: Topology, local: Scalar) -> SparseExtensor:
    """A form on each triangle's corners `[3, 3] [F]` gathered onto the vertices it joins,
    `[V <- V]`."""
    corners = topology.incidence[2, 0]                                        # [3] Scalar+[F <- V]
    return (~corners[:, None] * spdiag(local) * corners[None, :]).sum(axis=-1).sum(axis=-1)


# --- math -----------------------------------------------------------------------------
def triangles(topology: Topology, vertices: Point) -> tuple[Scalar, Plane, Scalar]:
    """Each triangle's area `Scalar+[F]`, each corner's hat gradient `[3] Plane+[F]`, the plane through the facing
    side and the condensate's own plane over twice the area, and each vertex's share of the area around it
    `Scalar+[V]`."""
    sides = topology.boundary[2, 1] * (topology.d[0] * vertices)[..., None]  # [3] Point-[F]
    # Twice a triangle's area: two of its sides joined with the condensate's own plane.
    double_areas = (sides[..., 0] & sides[..., 1]) & wave_plane               # Scalar-[F]
    areas = double_areas.norm() / 2                                           # Scalar+[F]
    # A corner's hat gradient: the plane through the facing side and the condensate's own plane, over twice the
    # area, so that it reads one from the side to the corner and nothing in the wave's plane.
    gradients = (wave_plane & sides) / double_areas[..., None]                # [3] Plane+[F]
    return areas, gradients, (~topology.incidence[2, 0] * (areas / 3)).sum(axis=-1)


def settling(topology: Topology, vertices: Point, areas: Scalar, gradients: Plane, cells: Scalar, trap: Scalar,
             repulsion: float, rotation: float, wave: Wave, step: float, n_steps: int, stride: int) -> Iterator[Wave]:
    """The condensate `Wave+[V]` every `stride` steps, from this one, descending its energy in the trap `trap`
    `Scalar+[V]` rotating at `rotation`, its atoms repelling with this strength, at unit atom number, on a mesh
    of triangles with these areas, hat gradients and cells: the kinetic energy, the trap and the rotation's drag taken
    implicitly, the repulsion explicitly."""
    corners = topology.incidence[2, 0]                                        # [3] Scalar+[F <- V]
    # The kinetic energy: each hat's gradient read against the flow another's drives, over the triangle.
    kinetic = (gradients[..., :, None] & mobility(gradients[..., None, :])) * areas[..., None, None] / 2
    # The rotation drags each triangle's middle a quarter turn in the trap's plane; the wave's change along the drag is
    # the drag paired with each corner's hat gradient, times the corner's wave, shared out to the corners' cells.
    drags = (corners.mean(axis=0) * vertices).commutator(mv.xy)               # Point-[F]
    along = (areas / 3)[..., None, None] * (drags[..., None, None] & gradients[..., None, :])
    # One implicit step, a map of waves factored once: the cells' wave over the step, the kinetic energy, the trap,
    # and the rotation's drag turned a quarter in the condensate's plane.
    stepping = (spdiag((cells / step + cells * trap) * Wave) + assembled(topology, kinetic * Wave) +
                assembled(topology, rotation * along * (Wave * mv.ab)))       # (Wave+ <- Wave+)[V <- V]
    for index in range(n_steps + 1):
        if index % stride == 0:
            yield wave
        density = wave.scalar_norm_squared()                                  # Scalar+[V]
        wave = stepping.solve(cells * (wave / step - repulsion * density * wave))
        # Back to unit atom number.
        wave = wave / np.sqrt((cells * wave.scalar_norm_squared()).sites.sum().to_array())


def windings(topology: Topology, wave: Wave) -> Turn:
    """Each triangle's winding `Turn+[F]`, in whole turns: the wave's turns along its edges, the logarithms of
    the rotors between the wave's directions at each edge's ends, added around it over a whole turn."""
    ends = topology.incidence[1, 0] * wave[..., None]                         # [2] Wave+[E]
    turns = (ends[..., 0] * ends[..., 1]).normalized().log()                  # Turn+[E]
    return (topology.boundary[2, 1] * turns[..., None]).sum(axis=-1) / (2 * np.pi)
