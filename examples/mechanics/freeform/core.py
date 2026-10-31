"""FreeForm elasticity: a mesh-free solid deformed by blending a few affine motions, in the geometric
algebra of three-dimensional space.

Points, displacements and the controls of the affine motions are directions, antivectors; the gradient
of a scalar field and the force on a point are covectors, vectors, read by pairing with a direction.
Weight fields over the solid blend whole affine motions, each set by four control displacements, and the
local deformation is the identity plus every control displacement paired with its feature's gradient,
`Antivector + state * (gradients & Antivector)`, a map on directions. Its Neo-Hookean stress maps
covectors to covectors: the metric part `F.adjoint().adjugate()`, and the cofactor, the deformation's
outermorphism onto the planes two directions span, which carries the plane spanned by two directions to
the plane spanned by their images, over the reference volume. The
pairing transpose of the sampling map, its adjugate, collects every local force onto the controls.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import VGA3D as ga
from examples import instantiate
from examples.mechanics.freeform.weights import kernels

mv = NumpyContext(ga).multivector
Mesh = instantiate("examples.util.mesh", ga).Mesh
tetrahedron = Mesh(4, [[0, 1, 2, 3]])
Scalar = ga.gatype.scalar()
# Points and displacements are antivectors; gradients and forces are covectors, vectors.
Antivector = ga.gatype.antivector()
Vector = ga.gatype.vector()
Deformation = ga.gatype((Antivector, Antivector))          # Antivector- <- Antivector-
# The stress, from gradients to forces.
Stress = ga.gatype((Vector, Vector))                        # Vector+ <- Vector+
Sampling = ga.gatype((Antivector, Antivector))
# The mass, from accelerations to forces.
Mass = ga.gatype((Vector, Antivector))                       # Vector+ <- Antivector-

# The reference volume, the join of the three axis directions: an intrinsic scalar of value one.
planes = stack((mv.x, mv.y, mv.z))                           # [3] Vector+
directions = planes.dual()                                   # [3] Antivector-
unit_volume = directions[0] & directions[1] & directions[2]
# Planes as the joins of the two directions that span them: vectors, oriented within themselves.
Span = ga.gatype.antibivector()


# --- math -----------------------------------------------------------------------------
def deformation_basis(queries: Antivector, centers: Antivector, radius: float,
                      modes: Scalar, corners: Antivector) -> tuple[Sampling, Vector]:
    """The displacement map and feature gradients for four affine controls per weight mode."""
    evaluation, gradient = kernels(queries, centers, radius)
    weights = evaluation[:, None](modes)                         # [samples, modes] Scalar+
    weight_gradients = gradient[:, None](modes)                   # [samples, modes] Vector+

    # Each control corner's barycentric gradient is its opposite outward face join over the tetrahedron's edge join.
    vertices = corners.field()                                   # Antivector-[corners]
    edges = (tetrahedron.oriented[3, 1] * (tetrahedron.d[0] * vertices)).batch()[:, 0]   # [tet_edges] Antivector-
    sides = tetrahedron.boundary[2, 1] * (tetrahedron.d[0] * vertices)                   # [face_corners] Antivector-[faces]
    face_patches = (tetrahedron.boundary[3, 2] * (sides[0] & sides[1])).batch()[:, 0]    # [4] Span-
    affine_gradients = -face_patches / (edges[0] & edges[1] & edges[2])                  # [4] Vector+
    affine_weights = 1 + (affine_gradients & (queries[:, None] - corners))  # [samples, 4] Scalar+

    features = weights[..., None] * affine_weights[:, None, :]   # [samples, modes, 4] Scalar+
    derivatives = (weight_gradients[..., None] * affine_weights[:, None, :]
                   + weights[..., None] * affine_gradients)     # [samples, modes, 4] Vector+
    # A state is the field of the four control displacements of every affine map.
    sampling = (features.reshape(len(queries), -1) * Antivector).field(1)  # [samples] Antivector- <- Antivector-[controls]
    gradients = derivatives.reshape(len(queries), -1).field()    # [samples] Vector+[controls]
    return sampling, gradients


def deformation(state: Antivector, gradients: Vector) -> Deformation:
    """The deformation of the blended affine motions: the identity plus each control displacement
    paired with its feature's gradient."""
    return Antivector + (state * (gradients & Antivector)).sites.sum()


def cofactor(local: Deformation) -> Stress:
    """The map on covectors that carries each plane, spanned by two directions, to the plane spanned by
    their images, over the reference volume: the deformation's outermorphism onto those joins, the
    determinant times the inverse transpose, singular or not."""
    # A plane is the join of two directions that span it, in units of the reference volume.
    return local.outermorphism(Span)(Vector * unit_volume) / unit_volume      # [...] Stress


def constitutive(local: Deformation, lame_lambda: float, lame_mu: float) -> tuple[Scalar, Stress]:
    """Stable Neo-Hookean energy density and its stress, from gradients to forces."""
    determinant = local.det()                                   # [samples] Scalar+
    bulk = lame_lambda + lame_mu
    target = 1 + lame_mu / bulk
    rest_energy = bulk * (1 - target)**2 + 3 * lame_mu
    # The stretch: the deformed metric read on reference directions, the Cauchy-Green form, traced against
    # the reference metric.
    stretch = local(Antivector).dot(local(Antivector)).contract()  # [samples] Scalar+
    energy = (bulk * (determinant - target).squared() + lame_mu * stretch - rest_energy) / 2
    # The stretch is metric, so its stress carries the deformation over to covectors through the metric.
    stress = lame_mu * local.adjoint().adjugate() + bulk * (determinant - target) * cofactor(local)   # [samples] Stress
    return energy, stress


def mass(sampling: Sampling, sample_masses: np.ndarray) -> Mass:
    """The control-space mass, from control accelerations to forces on the controls: each sample's
    displacement, turned into the covector the metric pairs with it, collected back onto the controls."""
    return sampling.field(0).adjugate()((sampling.dual() * sample_masses).field(0))


def potential(state: Antivector, sampling: Sampling, gradients: Vector,
              sample_volumes: np.ndarray, lame_lambda: float, lame_mu: float,
              clamp: Scalar, load: Vector) -> tuple[Scalar, Vector]:
    """Elastic energy, a compliant base clamp and an applied force density, with their gradient, the
    forces on the controls."""
    displacement = sampling(state)                              # [samples] Antivector-
    local = deformation(state, gradients)                        # [samples] Antivector- <- Antivector-
    density, stress = constitutive(local, lame_lambda, lame_mu)
    density = density + clamp * displacement.dot(displacement) / 2 - (load & displacement)
    energy = (density * sample_volumes).sum(axis=0)               # Scalar+
    elastic = (stress(gradients) * sample_volumes).sum(axis=0)     # Vector+[controls]
    # The clamp pulls back along the metric covector of the displacement; the adjugate of the sampling
    # collects every local force onto the affine controls.
    applied = sampling.field(0).adjugate()(
        ((clamp * displacement.dual() - load) * sample_volumes).field()
    )                                                           # Vector+[controls]
    return energy, elastic + applied
