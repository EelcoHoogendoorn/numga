"""Laplacian eigenfunction fluids evolve a thin incompressible fluid film over a curved triangulated surface through its
modal vorticity amplitudes.

The surface geometry determines the Laplace–Beltrami eigenmodes and their Lie–Poisson interaction coefficients once. On
each triangle, meeting the streamfunction's simplex gradient plane with the unit tangent patch gives the tangent fluid
velocity `gradient(stream) ^ normals`, and pairing another field's gradient plane with the area-weighted flow gives its
integrated advection rate.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np

from numga import NumpyContext
from numga.algebras import VGA3D as ga
from numga.sparse import spdiag
from examples import MODEL_DIR
from examples.util import shapes
from examples.util.topology import Topology

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                         # Scalar+
# Streamfunction simplex gradients on triangles are extrinsic planes orthogonal to the gradient direction.
Plane = ga.gatype.vector()                          # Vector+
# Vertex positions, edge displacements, and tangent fluid velocities are intrinsic antivectors.
Point = ga.gatype.antivector()                      # Antivector-
# Oriented triangle area elements and unit tangent patches are intrinsic joins of two edges.
Patch = ga.gatype.antibivector()                    # Antibivector-
# Vertex-integrated surface densities carry the intrinsic orientation of the pseudoscalar dual.
Antiscalar = ga.gatype.antiscalar()                 # Antiscalar-
GradientMap = ga.gatype((Plane, Scalar))
Mass = ga.gatype((Antiscalar, Scalar))
Superposition = ga.gatype((Scalar, Scalar))
Bracket = ga.gatype((Scalar, Scalar, Scalar))

# Path to the Stanford bunny model and the quarter turn standing its y axis upright along z.
bunny_path = MODEL_DIR / "bunny.obj"
upright = (mv.yz * (-np.pi / 4)).exp()              # [] Rotor+


# --- math -----------------------------------------------------------------------------
def structure_constants(topology: Topology, patches: Patch, gradient: GradientMap, modes: Scalar) -> Bracket:
    """Each modal Lie–Poisson structure constant integrates a mode against the surface Jacobian of two others, so
    applying the tensor to a modal streamfunction and modal vorticity gives every mode's advection rate."""
    # Averaging operator from triangle corners onto faces.
    corner_mean = topology.incidence[2, 0].mean(axis=0)                          # Scalar+[F <- V]
    face_gradients = gradient(modes)                                             # [n_modes] Plane+[F]
    # Meeting each mode's gradient with the triangle's area patch gives its area-weighted tangent flow,
    # which pairs with another gradient to integrate their Jacobian over the triangle.
    flows = face_gradients ^ patches                                             # [n_modes] Point-[F]
    jacobians = face_gradients[:, None] & flows[None, :]                         # [n_modes, n_modes] Scalar+[F]
    # Because the Jacobian is constant on each triangle, each mode integrates against it via its corner mean.
    face_means = corner_mean * modes                                             # [n_modes] Scalar+[F]
    integral = (face_means * Scalar).batch().field(0, 1)                         # (Scalar+ <- Scalar+)[n_modes <- F]
    # Total antisymmetry preserves cyclic permutations of the three mode indices, placing the vorticity mode last.
    return (integral(jacobians).batch() * Scalar * Scalar).field(0, 1, 2)        # (Scalar+ <- (Scalar+, Scalar+))[n_modes <- (n_modes, n_modes)]


def vorticity_rate(vorticity: Scalar, structure: Bracket, eigenvalues: Scalar, viscosity: float) -> Scalar:
    """The modal vorticity amplitudes change at the rate given by the Lie–Poisson advection bracket minus viscous
    decay."""
    stream = vorticity / eigenvalues                                             # Scalar+[n_modes]
    advection = structure(stream, vorticity)                                     # Scalar+[n_modes]
    return advection - viscosity * eigenvalues * vorticity                       # Scalar+[n_modes]


def advect(initial: Scalar, structure: Bracket, eigenvalues: Scalar, viscosity: float,
           dt: float, n_steps: int) -> Iterator[Scalar]:
    """Successive modal vorticity amplitudes advance through classical fourth-order Runge–Kutta steps."""
    vorticity = initial                                                          # Scalar+[n_modes]
    yield vorticity
    for _ in range(n_steps):
        first = vorticity_rate(vorticity, structure, eigenvalues, viscosity)     # Scalar+[n_modes]
        second = vorticity_rate(vorticity + (dt / 2) * first, structure, eigenvalues, viscosity)   # Scalar+[n_modes]
        third = vorticity_rate(vorticity + (dt / 2) * second, structure, eigenvalues, viscosity)   # Scalar+[n_modes]
        fourth = vorticity_rate(vorticity + dt * third, structure, eigenvalues, viscosity)   # Scalar+[n_modes]
        vorticity = vorticity + (dt / 6) * (first + 2 * second + 2 * third + fourth)   # Scalar+[n_modes]
        yield vorticity


def reconstruct(states: Iterable[Scalar], superposition: Superposition, eigenvalues: Scalar,
                gradient: GradientMap, normals: Patch) -> Iterator[tuple[Scalar, Scalar, Point]]:
    """Each modal vorticity state determines the vertex vorticity, vertex streamfunction, and face tangent velocity."""
    for amplitudes in states:
        vorticity = superposition(amplitudes)                                    # Scalar+[V]
        stream = superposition(amplitudes / eigenvalues)                         # Scalar+[V]
        velocity = gradient(stream) ^ normals                                    # Point-[F]
        yield vorticity, stream, velocity


def shear_layers(vertices: Point, dual_areas: Scalar, latitude: float, width: float,
                 strength: float, wave_amplitude: float) -> Scalar:
    """Opposing zonal vorticity bands on the unit sphere carry a smooth meridional ripple at zero net circulation."""
    heights = mv.z & vertices                                                    # Scalar+[V]
    n_meridians = 3
    angles = np.pi * np.arange(n_meridians) / n_meridians
    meridians = (mv.xy * (-angles / 2)).exp() >> mv.x                            # [n_meridians] Plane+
    first, second, third = meridians & vertices                                  # Scalar+[V], Scalar+[V], Scalar+[V]
    # Multiplying three meridian plane readings makes the azimuthal ripple vanish smoothly at both poles.
    shift = -4 * wave_amplitude * first * second * third                         # Scalar+[V]
    north = (heights - np.sin(latitude) - shift) / width                         # Scalar+[V]
    south = (heights + np.sin(latitude) - shift) / width                         # Scalar+[V]
    vorticity = strength * ((-0.5 * north.squared()).exp() - (-0.5 * south.squared()).exp())   # Scalar+[V]
    return vorticity - (dual_areas * vorticity).sites.sum() / dual_areas.sites.sum()


# --- plumbing -------------------------------------------------------------------------
def bunny(size: float) -> tuple[Topology, Point]:
    """The triangulated Stanford bunny stands upright along z, centred on its bounding box and scaled to longest side
    `size`."""
    topology, coordinates = shapes.surface(bunny_path, size)
    vertices = mv.vector(coordinates).dual().field()                             # Point-[V]
    return topology, upright >> vertices
