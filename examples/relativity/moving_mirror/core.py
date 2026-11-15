"""Scalar waves on a triangulated spacetime strip with a moving mirror boundary.

A spacetime triangle's signed patch defines corner interpolation and edge 1-forms from opposite side
displacements without invoking a metric. The Lorentzian signature enters once through the plane dual,
whose pullback through the simplex gradient forms the spacetime wave operator, while quiet initial rows
and boundary walls select the causal solution.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import Algebra, NumpyContext
from numga.sparse import Sparse, spdiag
from numga.topology import Topology

ga = Algebra("x+t-")
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                    # Scalar+
Plane = ga.gatype.vector()                                                     # Vector+
Point = ga.gatype.antivector()                                                 # Antivector-
Patch = ga.gatype.antibivector()                                               # Antibivector-
Amount = ga.gatype.antiscalar()                                                # Antiscalar-
WaveMap = ga.gatype((Amount, Scalar))                                          # Amount- <- Scalar+
GradientMap = ga.gatype((Plane, Scalar))                                       # Plane+ <- Scalar+
triangle = Topology(3, [[0, 1, 2]])

# Unit spatial and future-timelike displacements pair to one with `mv.x` and `mv.t`.
space, time = mv.x.dual(), -mv.t.dual()                                        # [] Point-, Point-


# --- math -----------------------------------------------------------------------------
def corner_interpolation(corners: Point, points: Point) -> tuple[Scalar, Plane, Patch]:
    """Dividing opposite side displacements of `corners` (`Point-[3]`) by twice the signed patch gives the simplex gradients (`[3] Plane+`), corner weights (`[3, ...] Scalar+`), and patch (`[] Patch-`)."""
    sides = (triangle.boundary[2, 1] * (triangle.d[0] * corners)).batch()[:, 0]  # [3] Point-
    patch = (sides[0] & sides[1]) / 2                                          # [] Patch-
    # Dividing each opposite worldline displacement by twice the signed patch gives a simplex gradient plane
    # that drops by one from its corner event to the opposite edge without using a metric.
    gradients = sides / (2 * patch)                                            # [3] Plane+
    corner_events = corners.batch()                                            # [3] Point-
    weights = 1 + (gradients[:, None] & (points[None, :] - corner_events[:, None]))  # [3, ...] Scalar+
    return weights, gradients, patch


def edge_whitney_forms(corners: Point, points: Point) -> Plane:
    """Dividing displacements from `corners` (`Point-[3]`) to `points` (`[...] Point-`) by twice the signed patch gives the three edge 1-forms (`[3, ...] Plane+`)."""
    sides = (triangle.boundary[2, 1] * (triangle.d[0] * corners)).batch()[:, 0]  # [3] Point-
    patch = (sides[0] & sides[1]) / 2                                          # [] Patch-
    corner_events = corners.batch()                                            # [3] Point-
    return (points[None, :] - corner_events[:, None]) / (2 * patch)            # [3, ...] Plane+


def spacetime_operators(topology: Topology, vertices: Point) -> tuple[GradientMap, WaveMap, Scalar]:
    """Assembling the simplex gradient across `topology` and pulling back the Lorentzian dual gives the gradient (`(Plane+ <- Scalar+)[F <- V]`), wave operator (`(Amount- <- Scalar+)[V <- V]`), and areas (`Scalar+[F]`)."""
    edges = topology.d[0] * vertices                                           # Point-[E]
    sides = topology.boundary[2, 1] * edges                                    # [3] Point-[F]
    patches = (sides[0] & sides[1]) / 2                                        # Patch-[F]
    areas = patches.norm()                                                     # Scalar+[F]
    corner_gradients = sides / (2 * patches)                                   # [3] Plane+[F]
    gradient = (spdiag(corner_gradients) * topology.incidence[2, 0]).sum(axis=0) * Scalar  # (Plane+ <- Scalar+)[F <- V]
    # Pulling the Lorentzian plane dual back through the simplex gradient builds a wave operator with null characteristics.
    medium = areas * Plane.dual()                                              # (Point- <- Plane+)[F]
    wave_operator = gradient.adjugate()(medium(gradient))                      # (Amount- <- Scalar+)[V <- V]
    return gradient, wave_operator, areas


def causal_wake(
    topology: Topology,
    vertices: Point,
    wave_operator: WaveMap,
    areas: Scalar,
    n_space: int,
    n_time: int,
    source_event: Point,
    source_width: float,
) -> Scalar:
    """Solving the spacetime wave equation with two quiet initial rows radiates the causal light-cone wake (`Scalar+[V]`) from `source_event`."""
    displacements = vertices - source_event                                    # Point-[V]
    delay = mv.t & displacements                                               # Scalar+[V]
    distance_squared = (mv.x & displacements).squared() + delay.squared()      # Scalar+[V]
    # A time-derivative Gaussian source radiates two sharp pulses along the forward 45-degree null rays.
    profile = (delay / source_width) * (-(distance_squared / (2 * source_width**2))).exp()  # Scalar+[V]
    source = (profile * areas.sites.mean()).dual()                             # Amount-[V]

    grid = np.arange(n_time * n_space).reshape(n_time, n_space)
    interior = topology.reading(grid[1:-1, 1:-1].ravel(), 0).adjugate() * Scalar  # (Scalar+ <- Scalar+)[V <- V_interior]
    future = topology.reading(grid[2:, 1:-1].ravel(), 0).adjugate() * Scalar   # (Scalar+ <- Scalar+)[V <- V_future]
    return future(interior.adjugate()(wave_operator(future)).solve(interior.adjugate()(source)))  # Scalar+[V]


def moving_mirror_wave(
    topology: Topology,
    vertices: Point,
    gradient: GradientMap,
    wave_operator: WaveMap,
    areas: Scalar,
    n_space: int,
    n_time: int,
    dt: float,
    pulse_centre: float,
    pulse_width: float,
    wavelength: float,
) -> tuple[Scalar, Plane, Scalar, Scalar]:
    """Solving for a right-moving wave packet reflecting off the moving boundary gives the potential (`Scalar+[V]`), gradient (`Plane+[F]`), nodal wave rate (`Scalar+[V]`), and slab energies (`[n_time - 1] Scalar+`)."""
    # Pairing with the null plane `mv.x - mv.t` reads the phase of the right-moving incident wave at every event.
    phase = ((mv.x - mv.t) & vertices) - pulse_centre                          # Scalar+[V]
    envelope = (-(phase / pulse_width).squared()).exp()                        # Scalar+[V]
    carrier = (2 * np.pi * phase / wavelength).cos()                           # Scalar+[V]

    grid = np.arange(n_time * n_space).reshape(n_time, n_space)
    seed = topology.reading(grid[:2, 1:-1].ravel(), 0)                         # Scalar+[V_seed <- V]
    initial = seed.adjugate() * (seed * (envelope * carrier))                  # Scalar+[V]

    interior = topology.reading(grid[1:-1, 1:-1].ravel(), 0).adjugate() * Scalar  # (Scalar+ <- Scalar+)[V <- V_interior]
    future = topology.reading(grid[2:, 1:-1].ravel(), 0).adjugate() * Scalar   # (Scalar+ <- Scalar+)[V <- V_future]
    rhs = -interior.adjugate()(wave_operator(initial))                         # Amount-[V_interior]
    potential = initial + future(interior.adjugate()(wave_operator(future)).solve(rhs))  # Scalar+[V]

    field = gradient(potential)                                                # Plane+[F]
    face_rate = field & time                                                   # Scalar+[F]
    to_vertices = topology.incidence[2, 0].sum(axis=0).adjugate()              # Scalar+[V <- F]
    wave_rate = (to_vertices * (areas * face_rate)) / (to_vertices * areas)    # Scalar+[V]

    # Reflecting the Lorentzian dual across the observer's slice flips the temporal sign into positive Hamiltonian energy density.
    energy_density = -(field & (mv.t >> Plane.dual()(field))) / 2              # Scalar+[F]
    slab_indices = topology.elements[2].min(axis=-1) // n_space                # [F]
    to_slabs = Sparse.selection(slab_indices, n_time - 1).adjugate()           # Scalar+[n_time - 1 <- F]
    slice_energies = (to_slabs * (areas * energy_density)).batch() / dt        # [n_time - 1] Scalar+
    return potential, field, wave_rate, slice_energies


def cavity_slices(n_time: int, n_frames: int) -> Iterator[int]:
    """Yields `n_frames` evenly spaced simultaneity row indices sweeping across the `n_time` worldsheet rows."""
    for row in np.linspace(4, n_time - 4, n_frames, dtype=int):
        yield int(row)
