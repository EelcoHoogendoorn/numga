"""A rotating Bose-Einstein condensate in a harmonic trap settles into a ground-state lattice of quantized vortices on
a triangulated disk, formulated in the product of the trap's plane and an internal phase plane of the condensate.

A condensate shares one quantum state across all of its atoms. Because its flow follows the spatial turning of that
wave rather than rigid-body rotation, circulation concentrates at quantized vortices, density holes around which the
wave turns once. Under steady rotation, vortices enter the cloud at Feynman's areal density of twice the rotation rate
over the circulation quantum, so the coarse-grained flow matches rigid-body rotation.

Each vertex value is a vector in the internal `a, b` plane whose norm is the square root of the atom density and whose
direction sets the local phase. Two bivectors govern the rotating frame: `xy` turns positions in the trap plane by a
quarter turn via `point.commutator(xy)`, while `ab` turns the wave by a quarter turn in its internal plane via
`wave * ab` and commutes with every spatial blade. Pairing the simplex gradients with the rotational drag velocity and
turning the wave by `ab` gives the rotating-frame energy term directly, and imaginary-time relaxation at unit atom
number settles the cloud into its vortex lattice. Summing the logarithms of edge rotors around each triangle yields its
integer winding number without angle unwrapping.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import Algebra, NumpyContext
from numga.sparse import SparseExtensor, spdiag
from numga.topology import Topology

# The trap's plane joined with the condensate's internal phase plane.
ga = Algebra("x+y+") * Algebra("a+b+")
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                   # Scalar+
Plane = ga.gatype.vector()                                                    # Plane+
Point = ga.gatype.antivector()                                                # Point-
Wave = Plane.derive.within(Algebra("a+b+"))                                   # Wave+
Turn = ga.gatype.bivector().derive.within(Algebra("a+b+"))                    # Turn+

# Unit displacements across and up the trap plane from its centre.
horizontal = mv.x.dual()                                                      # [] Point-
vertical = mv.y.dual()                                                        # [] Point-
# The condensate's internal plane as an intrinsic patch orthogonal to spatial displacements.
wave_plane = mv.a.dual() & mv.b.dual()                                        # [] Antibivector-
# Unit inverse-mass duality map turning a spatial gradient plane into the flow it drives.
mobility = Plane.dual()                                                       # [] Point- <- Plane+


# --- plumbing -------------------------------------------------------------------------
def corner_assembly(topology: Topology, local: Wave) -> SparseExtensor:
    """Each triangle's `[3, 3] (Wave+ <- Wave+)[F]` corner coupling pulls back across vertex-triangle incidence to a
    global `(Wave+ <- Wave+)[V <- V]` operator."""
    corners = topology.incidence[2, 0]                                        # [3] Scalar+[F <- V]
    return (corners[:, None].adjugate() * spdiag(local) * corners[None, :]).sum(axis=0).sum(axis=0)


# --- math -----------------------------------------------------------------------------
def triangle_geometry(topology: Topology, vertices: Point) -> tuple[Scalar, Plane, Scalar]:
    """Joining edge displacements with `wave_plane` determines each triangle's area (`Scalar+[F]`), its three corner
    simplex gradients (`[3] Plane+[F]`), and each vertex's lumped dual area (`Scalar+[V]`)."""
    corners = topology.incidence[2, 0]                                        # [3] Scalar+[F <- V]
    sides = topology.boundary[2, 1] * (topology.d[0] * vertices)              # [3] Point-[F]
    # Joining two edge displacements with the internal wave plane gives twice the signed triangle area.
    double_areas = (sides[0] & sides[1]) & wave_plane                         # Antiscalar-[F]
    areas = double_areas.antinorm() / 2                                       # Scalar+[F]
    # Joining the internal wave plane with each opposite edge and dividing by twice the area gives the simplex gradient.
    gradients = (wave_plane & sides) / double_areas                           # [3] Plane+[F]
    dual_areas = (corners.adjugate() * (areas / 3)).sum(axis=0)               # Scalar+[V]
    return areas, gradients, dual_areas


def relaxation_states(topology: Topology, vertices: Point, areas: Scalar, gradients: Plane, dual_areas: Scalar,
                      trap_potential: Scalar, repulsion: float, rotation: float, wave: Wave, step: float,
                      n_steps: int, stride: int) -> Iterator[Wave]:
    """Yields the condensate wave (`Wave+[V]`) every `stride` steps of imaginary-time relaxation at unit atom number,
    stepping kinetic, trap, and rotational-drag terms implicitly and contact repulsion explicitly."""
    corners = topology.incidence[2, 0]                                        # [3] Scalar+[F <- V]
    # Pairwise coupling of corner simplex gradients through the inverse-mass mobility over each triangle.
    kinetic = (gradients[:, None] & mobility(gradients[None, :])) * (areas / 2)   # [3, 3] Scalar+[F]
    # Quarter-turn velocity of each triangle centroid in the rotating frame, paired extrinsic-first with the gradients.
    drag_velocities = (corners.mean(axis=0) * vertices).commutator(mv.xy)     # Point-[F]
    advection = (areas / 3) * (gradients & drag_velocities)[None, :]          # [1, 3] Scalar+[F]
    local_operator = kinetic * Wave - rotation * advection * (Wave * mv.ab)   # [3, 3] (Wave+ <- Wave+)[F]
    step_operator = ((dual_areas / step + dual_areas * trap_potential) * Wave +
                     corner_assembly(topology, local_operator))               # (Wave+ <- Wave+)[V <- V]
    for index in range(n_steps + 1):
        if index % stride == 0:
            yield wave
        density = wave.scalar_norm_squared()                                  # Scalar+[V]
        unnormalized = step_operator.solve(dual_areas * (wave / step - repulsion * density * wave))   # Wave+[V]
        atom_count = (dual_areas * unnormalized.scalar_norm_squared()).sites.sum()                    # [] Scalar+
        wave = unnormalized / atom_count.square_root()                        # Wave+[V]


def triangle_windings(topology: Topology, wave: Wave) -> Turn:
    """Summing the logarithms of normalized edge rotors around each triangle boundary gives its phase winding
    (`Turn+[F]`) in whole turns."""
    ends = topology.incidence[1, 0] * wave                                    # [2] Wave+[E]
    edge_turns = (ends[0] * ends[1]).normalized().log()                       # Turn+[E]
    return (topology.boundary[2, 1] * edge_turns).sum(axis=0) / (2 * np.pi)   # Turn+[F]


def vortex_counts(topology: Topology, vertices: Point, wave: Wave, radii: np.ndarray,
                  vortex_threshold: float) -> np.ndarray:
    """Triangles whose phase winding exceeds `vortex_threshold` accumulate by centroid distance into enclosed vortex
    counts (`[n_radii]`) across `radii` (`[n_radii]`)."""
    centroids = topology.incidence[2, 0].mean(axis=0) * vertices              # Point-[F]
    turns = triangle_windings(topology, wave).scalar_product(~mv.ab)          # Scalar+[F]
    vortices = (turns > vortex_threshold) | (turns < -vortex_threshold)       # [F]
    return ((centroids.antinorm().batch() <= radii[:, None]) & vortices).sum(axis=-1)   # [n_radii]
