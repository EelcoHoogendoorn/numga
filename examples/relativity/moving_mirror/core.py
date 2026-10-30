"""A scalar pulse through a triangulated worldsheet with a moving boundary: an advancing mirror compresses the reflected pulse and supplies its extra energy.

The signed triangle patch defines the interpolation by corner hats and edge forms independently of the metric. The signature enters through `Plane.dual()`, and its pullback by the gradient gives the wave operator. Initial data and boundary values select the causal solution; the Lorentzian metric alone does not select a direction of time.
"""

from __future__ import annotations

import numpy as np

from numga import Algebra
from numga.sparse import SparseExtensor, spdiag
from examples import instantiate

mesh_module = instantiate("examples.util.mesh", Algebra("x+t-"))
ga, Mesh, mv = mesh_module.ga, mesh_module.Mesh, mesh_module.mv
Scalar = ga.gatype.scalar()                                                  # Scalar+
Plane = ga.gatype.vector()                                                   # Vector+
Point = ga.gatype.antivector()                                               # Antivector-
Patch = ga.gatype.antibivector()                                             # Antibivector-
Amount = ga.gatype.antiscalar()                                              # Antiscalar-

# Unit spatial and future-timelike displacements, which pair to one with `mv.x` and `mv.t`.
space, time = mv.x.dual(), -mv.t.dual()                                      # [] Point-, Point-


# --- math -----------------------------------------------------------------------------
def barycentric_hats(corners: Point, points: Point) -> tuple[Scalar, Plane, Patch]:
    """The barycentric hat values at `points`, the constant corner gradients, and the spacetime triangle's signed patch."""
    sides = corners[[2, 0, 1]] - corners[[1, 2, 0]]                          # [corners] Point-
    patch = (sides[0] & sides[1]) / 2                                        # [] Patch-
    # Dividing each opposite worldline displacement by twice the signed spacetime patch gives the
    # gradient plane of its corner's hat, which drops by one from its corner event to the opposite edge.
    gradients = sides / (2 * patch)                                          # [corners] Plane+
    hats = 1 + (gradients[:, None] & (points[None, :] - corners[:, None]))   # [corners, samples] Scalar+
    return hats, gradients, patch


def edge_whitney_forms(corners: Point, points: Point) -> Plane:
    """The three spacetime edge Whitney 1-forms evaluated at `points`, circulating around each edge's opposite corner."""
    sides = corners[[2, 0, 1]] - corners[[1, 2, 0]]                          # [corners] Point-
    patch = (sides[0] & sides[1]) / 2                                        # [] Patch-
    # Dividing the displacement from the opposite corner by twice the signed spacetime patch gives a
    # 1-form with unit line integral along its own edge and zero along the other two edges.
    return (points[None, :] - corners[:, None]) / (2 * patch)                # [corners, samples] Plane+


class Operators:
    """The gradient, the Lorentzian wave operator, and the face-to-vertex averaging on a spacetime mesh."""

    def __init__(self, mesh: Mesh, vertices: Point):
        self.mesh, self.vertices = mesh, vertices
        edges = mesh.d[0] * vertices                                         # Point-[edges]
        sides = mesh.boundary[2, 1] * edges                                  # [corners] Point-[faces]
        patches = (sides[0] & sides[1]) / 2                                  # Patch-[faces]
        self.areas = patches.norm()                                          # Scalar+[faces]

        # Each corner hat's gradient is constant across its spacetime triangle; assembling by corner
        # maps vertex potentials to piecewise-constant face 1-forms without any metric.
        corner_gradients = sides / (2 * patches)                             # [corners] Plane+[faces]
        self.gradient = (spdiag(corner_gradients) * mesh.incidence[2, 0]).sum(axis=0) * Scalar  # (Plane+ <- Scalar+)[faces <- vertices]

        # The Lorentzian dual supplies the timelike minus sign; pulling it back through
        # the gradient gives a wave operator with null characteristic directions.
        self.wave_operator = self.gradient.adjugate()(spdiag(self.areas * Plane.dual())(self.gradient))   # (Amount- <- Scalar+)[vertices <- vertices]

        # Area-weighted averaging from faces to vertices reconstructs smooth nodal fields for slicing.
        self.to_vertices = ~mesh.incidence[2, 0].sum(axis=0)                 # Scalar+[vertices <- faces]


def causal_wake(operators: Operators, n_space: int, n_time: int,
                source_event: Point, source_width: float) -> Scalar:
    """The causal light-cone wave wake radiated into the future of a localized spacetime source event."""
    displacements = operators.vertices.batch() - source_event                # [vertices] Point-
    delay = mv.t & displacements                                             # [vertices] Scalar+
    distance_squared = (mv.x & displacements).squared() + delay.squared()    # [vertices] Scalar+
    # A time-derivative Gaussian source radiates two sharp pulses along the forward 45-degree null rays.
    profile = (delay / source_width) * (-(distance_squared / (2 * source_width**2))).exp()   # [vertices] Scalar+
    source = (profile * operators.areas.sites.mean()).field().dual()         # Amount-[vertices]

    grid = np.arange(n_time * n_space).reshape(n_time, n_space)
    interior = ~SparseExtensor.selection(mv.context, grid[1:-1, 1:-1].ravel(), operators.mesh.n_vertices) * Scalar   # (Scalar+ <- Scalar+)[vertices <- interior]
    future = ~SparseExtensor.selection(mv.context, grid[2:, 1:-1].ravel(), operators.mesh.n_vertices) * Scalar   # (Scalar+ <- Scalar+)[vertices <- future]
    # The first two rows and the walls stay zero; each interior equation fixes the next time row.
    return future(interior.adjugate()(operators.wave_operator(future)).solve(interior.adjugate()(source)))   # Scalar+[vertices]


def moving_mirror_wave(operators: Operators, n_space: int, n_time: int, dt: float,
                       pulse_centre: float, pulse_width: float, wavelength: float
                       ) -> tuple[Scalar, Plane, Scalar, Scalar]:
    """The scalar potential, spacetime gradient, time derivative, and slice energies of a pulse reflecting off a moving mirror."""
    mesh, vertices = operators.mesh, operators.vertices
    # Pairing with the null plane `mv.x - mv.t` reads the phase of the right-moving wave at every event.
    phase = ((mv.x - mv.t) & vertices).batch() - pulse_centre                # [vertices] Scalar+
    envelope = (-(phase / pulse_width).squared()).exp()                      # [vertices] Scalar+
    carrier = (2 * np.pi * phase / wavelength).cos()                         # [vertices] Scalar+

    grid = np.arange(n_time * n_space).reshape(n_time, n_space)
    seed = SparseExtensor.selection(mv.context, grid[:2, 1:-1].ravel(), mesh.n_vertices)   # Scalar+[seed <- vertices]
    initial = ~seed * (seed * (envelope * carrier).field())                  # Scalar+[vertices]

    interior = ~SparseExtensor.selection(mv.context, grid[1:-1, 1:-1].ravel(), mesh.n_vertices) * Scalar   # (Scalar+ <- Scalar+)[vertices <- interior]
    future = ~SparseExtensor.selection(mv.context, grid[2:, 1:-1].ravel(), mesh.n_vertices) * Scalar   # (Scalar+ <- Scalar+)[vertices <- future]
    rhs = -interior.adjugate()(operators.wave_operator(initial))             # Amount-[interior]
    potential = initial + future(interior.adjugate()(operators.wave_operator(future)).solve(rhs))   # Scalar+[vertices]

    field = operators.gradient(potential)                                    # Plane+[faces]
    face_rate = field & time                                                 # Scalar+[faces]
    wave_rate = (operators.to_vertices * (operators.areas * face_rate)) / (operators.to_vertices * operators.areas)   # Scalar+[vertices]

    # Reflecting the Lorentzian dual across the observer's slice mv.t flips the temporal sign,
    # turning the Lagrangian density into the positive-definite Hamiltonian wave energy density.
    energy_density = -(field & (mv.t >> Plane.dual()(field))) / 2            # Scalar+[faces]
    # Each face lies in the time slab of its earliest corner; the reverse of reading each
    # face's slab sums the faces into the slabs.
    slab_indices = mesh.faces.min(axis=-1) // n_space                        # [faces]
    to_slabs = ~SparseExtensor.selection(mv.context, slab_indices, n_time - 1)   # Scalar+[layers <- faces]
    slice_energies = (to_slabs * (operators.areas * energy_density)).batch() / dt   # [layers] Scalar+
    return potential, field, wave_rate, slice_energies
