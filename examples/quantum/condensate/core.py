"""A rotating Bose-Einstein condensate in an elliptical harmonic trap evolves in real time under the damped
Gross-Pitaevskii equation on a triangulated disk, formulated in the product of the trap's plane and an internal phase
plane of the condensate.

A condensate shares one quantum state across all of its atoms. Because its superfluid flow follows the spatial turning
of that wave rather than rigid-body rotation, circulation enters from the rim as quantized vortices, density holes
around which the wave turns once. Stirring the cloud with a mildly elliptical rotating trap in the presence of thermal
dissipation draws vortices into the interior, where they arrange into a triangular lattice at Feynman's areal density
of twice the rotation rate over the circulation quantum.

Each vertex value is modelled as a vector in the internal `a, b` plane whose norm is the square root of the atom
density and whose direction sets the local phase, with no imaginary unit. Two commuting bivectors govern the rotating
frame. The trap plane `xy` turns positions by a quarter turn via `point.commutator(xy)` into rotational drag velocity,
while the internal plane `ab` turns the wave by a quarter turn via `wave * ab`. A single even phase-response element
`(damping - ab) / (1 + damping ** 2)` combines real-time phase precession and thermal dissipation, and subtracting the
instantaneous chemical potential preserves unit atom number while eliminating global phase precession. Summing the
logarithms of normalized edge rotors around each triangle yields its exact integer winding number without angle
unwrapping.
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
Plane = ga.gatype.vector()                                                    # Vector+
Point = ga.gatype.antivector()                                                # Antivector-
Wave = Plane.derive.within(Algebra("a+b+"))                                   # Vector+
Turn = ga.gatype.bivector().derive.within(Algebra("a+b+"))                    # Bivector+

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


def condensate_states(
    topology: Topology,
    vertices: Point,
    areas: Scalar,
    gradients: Plane,
    dual_areas: Scalar,
    trap_potential: Scalar,
    repulsion: float,
    rotation: float,
    damping: float,
    wave: Wave,
    time_step: float,
    n_steps: int,
    stride: int,
) -> Iterator[Wave]:
    """Yields the condensate wave (`Wave+[V]`) every `stride` real-time steps of the damped rotating-frame
    Gross-Pitaevskii equation at unit atom number, advancing kinetic, trap, and rotational-drag terms by
    Crank-Nicolson and contact repulsion less the instantaneous chemical potential explicitly."""
    corners = topology.incidence[2, 0]                                        # [3] Scalar+[F <- V]
    # Pairwise coupling of corner simplex gradients through the inverse-mass mobility over each triangle.
    kinetic = (gradients[:, None] & mobility(gradients[None, :])) * (areas / 2)   # [3, 3] Scalar+[F]
    # Quarter-turn velocity of each triangle centroid in the rotating frame, paired extrinsic-first with the gradients.
    drag_velocities = (corners.mean(axis=0) * vertices).commutator(mv.xy)     # Point-[F]
    advection = (areas / 3) * (gradients & drag_velocities)[None, :]          # [1, 3] Scalar+[F]
    local_energy = kinetic * Wave - rotation * advection * (Wave * mv.ab)     # [3, 3] (Wave+ <- Wave+)[F]
    linear_energy = dual_areas * trap_potential * Wave + corner_assembly(topology, local_energy)   # (Wave+ <- Wave+)[V <- V]
    # Single even phase-response element combining thermal dissipation and real-time phase precession.
    phase_response = (damping - mv.ab) / (1 + damping ** 2)                   # [] Even+
    energy_rate = (dual_areas * trap_potential * (Wave * phase_response) +
                   corner_assembly(topology, local_energy * phase_response))   # (Wave+ <- Wave+)[V <- V]
    mass = dual_areas / time_step * Wave                                      # (Wave+ <- Wave+)[V <- V]
    implicit = mass + 0.5 * energy_rate                                       # (Wave+ <- Wave+)[V <- V]
    explicit = mass - 0.5 * energy_rate                                       # (Wave+ <- Wave+)[V <- V]
    yield wave
    for index in range(1, n_steps + 1):
        density = wave.scalar_norm_squared()                                  # Scalar+[V]
        # Subtracting the instantaneous chemical potential preserves atom number and removes global phase precession.
        energy_wave = linear_energy(wave) + dual_areas * repulsion * density * wave   # Wave+[V]
        chemical_potential = wave.scalar_product(energy_wave).sites.sum()     # [] Scalar+
        forcing = explicit(wave) - dual_areas * (repulsion * density - chemical_potential) * (wave * phase_response)   # Wave+[V]
        unnormalized = implicit.solve(forcing)                                # Wave+[V]
        atom_count = (dual_areas * unnormalized.scalar_norm_squared()).sites.sum()   # [] Scalar+
        wave = unnormalized / atom_count.square_root()                        # Wave+[V]
        if index % stride == 0:
            yield wave


def triangle_windings(topology: Topology, wave: Wave) -> Turn:
    """The logarithms of normalized edge rotors summed around each triangle boundary give its phase winding
    (`Turn+[F]`) in whole turns."""
    tail, head = topology.incidence[1, 0] * wave                              # Wave+[E], Wave+[E]
    edge_turns = (tail * head).normalized().log()                             # Turn+[E]
    return (topology.boundary[2, 1] * edge_turns).sum(axis=0) / (2 * np.pi)   # Turn+[F]


def vortex_counts(topology: Topology, vertices: Point, wave: Wave, radii: np.ndarray,
                  vortex_threshold: float) -> np.ndarray:
    """Triangles whose phase winding exceeds `vortex_threshold` accumulate by centroid distance into enclosed vortex
    counts (`[n_radii]`) across `radii` (`[n_radii]`)."""
    centroids = topology.incidence[2, 0].mean(axis=0) * vertices              # Point-[F]
    turns = triangle_windings(topology, wave).scalar_product(~mv.ab)          # Scalar+[F]
    vortices = (turns > vortex_threshold) | (turns < -vortex_threshold)       # [F]
    return ((centroids.antinorm().batch() <= radii[:, None]) & vortices).sum(axis=-1)   # [n_radii]
