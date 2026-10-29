"""Whitney displacement fields carry elastic waves through a planet with a liquid core and a free surface.

Corner hats give a constant strain on each triangle. The material reads that strain,
and its response pulls back to nodal forces; a lumped mass map turns them into motion.
Zero shear stiffness gives a liquid; its shear viscosity, the shear part of the same work form applied to the
velocity, damps the shear motion it cannot store. Liquid and solid share the vertices where they meet.
The divergence and the rotation of the velocity, the corner gradients paired with and wedged against the corner
velocities, separate compressional from shear motion.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np

from numga.algebras import VGA2D as ga
from numga.sparse import SparseExtensor, spdiag
from examples import instantiate

mesh_module = instantiate("examples.util.mesh", ga)
Mesh, as_scalar, mv = mesh_module.Mesh, mesh_module.as_scalar, mesh_module.mv
Scalar = ga.gatype.scalar()                                      # Scalar+
Point = ga.gatype.antivector()                                   # Antivector-
Plane = ga.gatype.vector()                                       # Vector+
Force = ga.gatype.vector()                                       # Vector+
Elastic = ga.gatype((Scalar, Plane, Plane, Point, Point))
Stiffness = ga.gatype((Force, Point))


dilation_traction = Plane * (Plane & Point)                    # [] Plane+ <- (Plane+, Plane+, Point-)
# The exterior-product term removes rotation from the gradient, leaving the two symmetric shear readings.
direct_shear = (Plane | Plane) * Point.dual_inverse()
shear_traction = 2 * direct_shear - (Plane | (Plane ^ Point.dual_inverse()))   # [] Plane+ <- (Plane+, Plane+, Point-)


# --- math -----------------------------------------------------------------------------
def constitutive(lame_lambda: Scalar, lame_mu: Scalar) -> Elastic:
    """A plane-strain isotropic elastic work form on pairs of displacement gradients."""
    traction = lame_lambda * dilation_traction + lame_mu * shear_traction
    return traction & Point                                    # [...] Scalar+ <- (Plane+, Plane+, Point-, Point-)


def viscous(viscosity: Scalar) -> Elastic:
    """A shear viscosity as a work form on pairs of velocity gradients: the elastic form's shear part less two thirds
    of its dilation, the stress of a liquid with no bulk viscosity in a plane-strain slice of three-dimensional space,
    so that it resists the rate of shear and not the rate of swelling."""
    return (viscosity * (shear_traction - 2 * dilation_traction / 3)) & Point   # [...] Scalar+ <- (Plane+, Plane+, Point-, Point-)


class Operators:
    """The planet's assembled elastic and viscous response and inverse inertia, shared by every time step."""

    def __init__(self, mesh: Mesh, vertices: Point, material: Elastic, viscosity: Elastic, density: Scalar):
        self.mesh, self.vertices = mesh, vertices
        sides = mesh.boundary[2, 1] * (mesh.d[0] * vertices)    # [corners] Point-[faces]
        patches = (sides[0] & sides[1]) / 2                     # Patch-[faces]
        self.areas = patches.norm()                             # Scalar+[faces]
        self.corner_gradients = sides / (2 * patches)           # [corners] Plane+[faces]
        self.stiffness = self.assemble(material)                # [...] (Force+ <- Point-)[vertices <- vertices]
        self.damping = self.assemble(viscosity)                 # [...] (Force+ <- Point-)[vertices <- vertices]

        # Each affine corner hat owns one third of its triangle's mass.
        corner_mean = mesh.incidence[2, 0].mean(axis=0)         # Scalar+[faces <- vertices]
        self.masses = ~corner_mean * (density * self.areas)     # [...] Scalar+[vertices]
        inverse_inertia = spdiag((1 / self.masses) * Force.dual())
        self.acceleration = -inverse_inertia(self.stiffness)    # [...] (Point- <- Point-)[vertices <- vertices]
        self.drag = -inverse_inertia(self.damping)              # [...] (Point- <- Point-)[vertices <- vertices]

        # Preassembled corner-hat operators read the rate of dilation and rotation directly from nodal velocities.
        self.divergence = (spdiag(self.corner_gradients & Point) * mesh.incidence[2, 0]).sum(axis=0)               # (Scalar+ <- Point-)[faces <- vertices]
        self.rotation = (spdiag(self.corner_gradients ^ Point.dual_inverse()) * mesh.incidence[2, 0]).sum(axis=0)  # (Bivector+ <- Point-)[faces <- vertices]

    def assemble(self, material: Elastic) -> Stiffness:
        """A work form on pairs of gradients, bound to each triangle's test and trial corner gradients: a form on the
        corner motions, summed over the triangles."""
        gradients = self.corner_gradients.batch()               # [corners, faces] Plane+
        work = material[None, None](gradients[:, None], gradients[None, :], Point, Point)
        local = (Force & Point).solve(work) * self.areas.batch()
        return SparseExtensor.from_indices(local, self.mesh.faces.T[:, None, :], self.mesh.faces.T[None, :, :],
                                           (self.mesh.n_vertices, self.mesh.n_vertices))


def evolve(operators: Operators, displacement: Point, velocity: Point, damping_time: float,
           dt: float, n_frames: int, n_substeps: int) -> Iterator[tuple[Point, Point]]:
    """Velocity Verlet advances the free elastic field, yielding displacement and velocity at frame times.

    The material is viscoelastic: it responds to its strain plus the damping time times its rate of strain, so that
    a mode of angular frequency `omega` loses amplitude at the rate `damping_time * omega**2 / 2`; the liquid's
    viscosity adds a drag on its rate of shear.
    """
    acceleration = operators.acceleration(displacement + damping_time * velocity) + operators.drag(velocity)   # [...] Point-[vertices]
    yield displacement, velocity
    for _ in range(n_frames):
        for _ in range(n_substeps):
            velocity = velocity + (dt / 2) * acceleration
            displacement = displacement + dt * velocity
            # The rate of strain is read from the half-step velocity, which keeps the step explicit.
            acceleration = operators.acceleration(displacement + damping_time * velocity) + operators.drag(velocity)
            velocity = velocity + (dt / 2) * acceleration
        yield displacement, velocity


def reconstruct(operators: Operators, states: Iterable[tuple[Point, Point]]
                ) -> Iterator[tuple[Point, Scalar, Scalar]]:
    """The deformed planet, the rate of dilation, and the signed vorticity at each frame."""
    for displacement, velocity in states:
        # Each corner gradient reads its corner's velocity: their sum is the rate at which the triangle swells.
        dilation_rate = operators.divergence(velocity)          # [...] Scalar+[faces]
        vorticity = operators.rotation(velocity) & mv.xy.dual() # [...] Scalar+[faces]
        yield displacement, dilation_rate, vorticity


def normal_modes(operators: Operators, count: int) -> tuple[Scalar, Point]:
    """The count lowest normal modes: their angular frequencies squared `[count]` and their shapes `[count]`,
    orthonormal in the masses. The stiffness, a map from displacements to forces, becomes a map of displacements to
    themselves through the dual that takes a force to a point, and so meets the masses on the same footing."""
    to_points = spdiag(as_scalar(np.ones(operators.mesh.n_vertices)).field() * Force.dual())
    return to_points(operators.stiffness).eigh(spdiag(operators.masses * Point), count)


def travelling(operators: Operators, first: Point, second: Point, peak: float, n_cycle_frames: int
               ) -> Iterator[tuple[Point, Scalar, Scalar]]:
    """The wave that each pair of modes of one frequency `[pairs]` makes, the first swinging a quarter cycle ahead of the
    second, through one cycle, in which it travels one wavelength round the ball; its largest displacement `peak`,
    with the dilation and the signed rotation it makes."""
    for pair in range(len(first)):
        sizes = first[pair].norm().batch()                                      # [vertices] Scalar+
        scale = peak / sizes[sizes.argmax()]                                    # [] Scalar+
        for phase in np.linspace(0, 2 * np.pi, n_cycle_frames, endpoint=False):
            displacement = scale * (np.cos(phase) * first[pair] + np.sin(phase) * second[pair])   # Point-[vertices]
            yield displacement, operators.divergence(displacement), operators.rotation(displacement) & mv.xy.dual()


def energy(operators: Operators, displacement: Point, velocity: Point) -> Scalar:
    """Kinetic plus elastic energy of the discrete field."""
    kinetic = (operators.masses * (velocity.dual_inverse() & velocity)).sites.sum() / 2
    elastic = (operators.stiffness(displacement) & displacement).sites.sum() / 2
    return kinetic + elastic                                   # [...] Scalar+
