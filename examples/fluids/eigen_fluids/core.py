"""Laplacian eigenfunction fluids evolve a surface flow through its modal amplitudes.

The surface geometry determines the Laplace–Beltrami modes and their interaction coefficients. A streamfunction drives the tangent flow `gradient(stream) ^ normals`; pairing a field's gradient with that flow gives its transport rate.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np

from numga import stack
from numga.algebras import VGA3D as ga
from numga.sparse import spdiag
from examples import MODEL_DIR, instantiate

mesh_module = instantiate("examples.util.mesh", ga)
Mesh, as_scalar, context = mesh_module.Mesh, mesh_module.as_scalar, mesh_module.ctx
mv = context.multivector
Scalar = ga.gatype.scalar()                         # Scalar+
# Streamfunction gradients on triangles are extrinsic planes across the gradient direction.
Plane = ga.gatype.vector()                          # Vector+
# Vertex positions, edge displacements, and tangent fluid velocities are intrinsic antivectors.
Point = ga.gatype.antivector()                      # Antivector-
# Oriented triangle area elements and unit tangent patches are joins of two edges.
Patch = ga.gatype.antibivector()                    # Antibivector-
GradientMap = ga.gatype((Plane, Scalar))
Mass = ga.gatype((ga.gatype.antiscalar(), Scalar))
Superposition = ga.gatype((Scalar, Scalar))
Bracket = ga.gatype((Scalar, Scalar, Scalar))

# The Stanford bunny from the shared models, and the quarter turn that stands its y axis upright along z.
bunny_path = MODEL_DIR / "bunny.obj"
upright = (mv.yz * (-np.pi / 4)).exp()              # [] Rotor+


# --- math -----------------------------------------------------------------------------
def structure_constants(mesh: Mesh, patches: Patch, gradient: GradientMap, modes: Scalar) -> Bracket:
    """Each modal Lie–Poisson structure constant is the surface integral of a mode against the Jacobian of two
    others, the wedge of their gradients read on the surface. Bound to a streamfunction and a vorticity, the
    constants give the advection rate of every mode."""
    thirds = mesh.incidence[2, 0].mean(axis=0)                                   # Scalar+[faces <- vertices]
    face_gradients = gradient(modes)                                             # [modes] Plane+[faces]
    # Meeting each mode's gradient with the triangle's area patch gives its area-weighted tangent velocity,
    # which pairs against another gradient to give the Jacobian integrated over the triangle.
    flows = face_gradients ^ patches                                             # [modes] Point-[faces]
    jacobians = face_gradients[:, None] & flows[None, :]                         # [modes, modes] Scalar+[faces]
    # Against a Jacobian constant on each triangle, a mode integrates through its mean over the triangle's corners.
    face_means = thirds * modes                                                  # [modes] Scalar+[faces]
    integral = (face_means * Scalar).batch().field(0, 1)                         # (Scalar+ <- Scalar+)[modes <- faces]
    # Totally antisymmetric, the constants are unchanged by a cyclic shift of their three modes: the mode the
    # Jacobians are integrated against may stand last, as the mode of the vorticity.
    return (integral(jacobians).batch() * Scalar * Scalar).field(0, 1, 2)       # (Scalar+ <- (Scalar+, Scalar+))[modes <- (modes, modes)]


def vorticity_rate(vorticity: Scalar, structure: Bracket, eigenvalues: Scalar, viscosity: float) -> Scalar:
    """The modal vorticity amplitudes change at the rate of the Lie–Poisson advection bracket of streamfunction
    and vorticity, less viscous diffusion."""
    stream = vorticity / eigenvalues                                             # Scalar+[modes]
    advection = structure(stream, vorticity)                                     # Scalar+[modes]
    return advection - viscosity * eigenvalues * vorticity                       # Scalar+[modes]


def advect(initial: Scalar, structure: Bracket, eigenvalues: Scalar, viscosity: float,
           dt: float, n_steps: int) -> Iterator[Scalar]:
    """Successive modal vorticities follow classical fourth-order Runge–Kutta steps."""
    vorticity = initial                                                          # Scalar+[modes]
    yield vorticity
    for _ in range(n_steps):
        first = vorticity_rate(vorticity, structure, eigenvalues, viscosity)                       # Scalar+[modes]
        second = vorticity_rate(vorticity + (dt / 2) * first, structure, eigenvalues, viscosity)   # Scalar+[modes]
        third = vorticity_rate(vorticity + (dt / 2) * second, structure, eigenvalues, viscosity)   # Scalar+[modes]
        fourth = vorticity_rate(vorticity + dt * third, structure, eigenvalues, viscosity)         # Scalar+[modes]
        vorticity = vorticity + (dt / 6) * (first + 2 * second + 2 * third + fourth)               # Scalar+[modes]
        yield vorticity


def reconstruct(states: Iterable[Scalar], superposition: Superposition, eigenvalues: Scalar,
                gradient: GradientMap, normals: Patch) -> Iterator[tuple[Scalar, Scalar, Point]]:
    """Each modal state gives the surface vorticity, streamfunction, and tangent velocity of one frame."""
    for amplitudes in states:
        vorticity = superposition(amplitudes)                                    # Scalar+[vertices]
        stream = superposition(amplitudes / eigenvalues)                         # Scalar+[vertices]
        velocity = gradient(stream) ^ normals                                    # Point-[faces]
        yield vorticity, stream, velocity


def shear_layers(vertices: Point, dual_areas: Scalar, latitude: float, width: float,
                 strength: float, wave_amplitude: float) -> Scalar:
    """Opposite vorticity bands with a meridional ripple and zero total circulation."""
    points = vertices.batch()                                                    # [vertices] Point-
    heights = mv.z & points                                                      # [vertices] Scalar+
    n_meridians = 3
    angles = np.pi * np.arange(n_meridians) / n_meridians
    meridians = (mv.xy * (-angles / 2)).exp() >> mv.x                            # [meridians] Plane+
    first, second, third = meridians[:, None] & points                           # [vertices] Scalar+, Scalar+, Scalar+
    # Three meridian readings make the azimuthal ripple vanish smoothly at the poles.
    shift = -4 * wave_amplitude * first * second * third                         # [vertices] Scalar+
    north = (heights - np.sin(latitude) - shift) / width                         # [vertices] Scalar+
    south = (heights + np.sin(latitude) - shift) / width                         # [vertices] Scalar+
    vorticity = (strength * ((-0.5 * north.squared()).exp() - (-0.5 * south.squared()).exp())).field()   # Scalar+[vertices]
    return vorticity - (dual_areas * vorticity).sites.sum() / dual_areas.sites.sum()


# --- plumbing -------------------------------------------------------------------------
def bunny(size: float) -> tuple[Mesh, Point]:
    """The Stanford bunny from the shared model repository, stood upright with z up, centred on its bounding
    box and scaled so that the box's longest side is `size`."""
    mesh, vertices = Mesh.read(bunny_path, size)
    return mesh, upright >> vertices
