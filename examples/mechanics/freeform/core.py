"""FreeForm elasticity deforms a solid without a volume discretisation by blending a few affine motions
in the geometric algebra of three-dimensional space.

Points, displacements, and the controls of the affine motions are directions (`Antivector-`), whereas
spatial gradients and forces are covectors (`Vector+`), paired with directions via `&`. Smooth weight
fields over the solid blend whole affine motions, each set by four control displacements, so the local
deformation is `Antivector + (state * (gradients & Antivector)).sites.sum()`, a map on directions. Its
Neo-Hookean stress maps covectors to covectors through the metric part `local.adjoint().adjugate()` and
the cofactor, the deformation's outermorphism onto planes spanned by two directions over the reference
volume. Pulling back across `&` via `.adjugate()` collects every sampled force onto the controls.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import VGA3D as ga
from examples.mechanics.freeform.weights import kernels
from numga.topology import Topology

mv = NumpyContext(ga, units=True).multivector
tetrahedron = Topology(4, [[0, 1, 2, 3]])
Scalar = ga.gatype.scalar()
# Points and displacements are antivectors; gradients and forces are covectors, vectors.
Antivector = ga.gatype.antivector()
Vector = ga.gatype.vector()
Deformation = ga.gatype((Antivector, Antivector))            # Antivector- <- Antivector-
# The stress maps spatial gradients to forces.
Stress = ga.gatype((Vector, Vector))                         # Vector+ <- Vector+
Sampling = ga.gatype((Antivector, Antivector))               # Antivector- <- Antivector-
# The control-space mass maps accelerations to forces.
Mass = ga.gatype((Vector, Antivector))                       # Vector+ <- Antivector-

# The reference volume is the join of the three axis directions: an intrinsic scalar of value one.
planes = stack((mv.x, mv.y, mv.z))                           # [3] Vector+
directions = planes.dual()                                   # [3] Antivector-
unit_volume = directions[0] & directions[1] & directions[2]  # Antiscalar-
# Planes as the joins of the two directions that span them, oriented within themselves.
Span = ga.gatype.antibivector()


# --- math -----------------------------------------------------------------------------
def deformation_basis(queries: Antivector, centers: Antivector, radius: Scalar,
                      modes: Scalar, corners: Antivector) -> tuple[Sampling, Vector]:
    """Blending four affine control corners per scalar mode yields the displacement sampling map and feature gradients at `queries`."""
    evaluation, gradient = kernels(queries, centers, radius)
    weights = evaluation[:, None](modes)                         # [..., ...] Scalar+
    weight_gradients = gradient[:, None](modes)                  # [..., ...] Vector+

    # Each control corner's simplex gradient is its opposite outward face join divided by the tetrahedron's edge join.
    vertices = corners.field()                                   # Antivector-[4]
    edges = (tetrahedron.oriented[3, 1] * (tetrahedron.d[0] * vertices)).batch()[:, 0]   # [3] Antivector-
    sides = tetrahedron.boundary[2, 1] * (tetrahedron.d[0] * vertices)                   # [3] Antivector-[4]
    face_patches = (tetrahedron.boundary[3, 2] * (sides[0] & sides[1])).batch()[:, 0]    # [4] Span-
    affine_gradients = -face_patches / (edges[0] & edges[1] & edges[2])                  # [4] Vector+
    affine_weights = 1 + (affine_gradients & (queries[:, None] - corners))  # [..., 4] Scalar+

    features = weights[..., None] * affine_weights[:, None, :]   # [..., ..., 4] Scalar+
    derivatives = (weight_gradients[..., None] * affine_weights[:, None, :]
                   + weights[..., None] * affine_gradients)      # [..., ..., 4] Vector+
    # A state is the field of the four control displacements of every affine map.
    sampling = (features.reshape(len(queries), -1) * Antivector).field(1)  # [...] (Antivector- <- Antivector-)[1 <- ...]
    gradients = derivatives.reshape(len(queries), -1).field()    # [...] Vector+[...]
    return sampling, gradients


def deformation(state: Antivector, gradients: Vector) -> Deformation:
    """Blending the control displacements `state` with `gradients` forms the local deformation map from reference directions to deformed directions."""
    return Antivector + (state * (gradients & Antivector)).sites.sum()


def cofactor(local: Deformation) -> Stress:
    """The outermorphism of `local` onto two-direction joins carries each reference plane to its deformed image per unit volume, giving the cofactor map on covectors even at flat collapse."""
    # A plane is the join of two directions that span it, measured in units of the reference volume.
    return local.outermorphism(Span)(Vector * unit_volume) / unit_volume      # [...] Vector+ <- Vector+


def constitutive(local: Deformation, lame_lambda: Scalar, lame_mu: Scalar) -> tuple[Scalar, Stress]:
    """Stable Neo-Hookean elasticity assigns an energy density (`[...] Scalar+`) and first Piola-Kirchhoff stress (`[...] Vector+ <- Vector+`) to `local`."""
    determinant = local.det()                                    # [...] Scalar+
    bulk = lame_lambda + lame_mu
    target = 1 + lame_mu / bulk
    rest_energy = bulk * (1 - target)**2 + 3 * lame_mu
    # The stretch traces the deformed metric against the reference metric across the three reference directions.
    stretch = local.symmetric_reverse_product().contract()       # [...] Scalar+
    energy = (bulk * (determinant - target).squared() + lame_mu * stretch - rest_energy) / 2
    # The stretch acts through the metric while the volume term acts through the cofactor.
    stress = lame_mu * local.adjoint().adjugate() + bulk * (determinant - target) * cofactor(local)   # [...] Vector+ <- Vector+
    return energy, stress


def control_mass(sampling: Sampling, sample_masses: Scalar) -> Mass:
    """Pulling each sample's metric displacement covector back through `sampling` assembles the control-space mass map from accelerations to forces."""
    return sampling.field(0).adjugate()((sampling.dual() * sample_masses).field(0))


def potential(state: Antivector, sampling: Sampling, gradients: Vector,
              sample_volumes: Scalar, lame_lambda: Scalar, lame_mu: Scalar,
              clamp: Scalar, load: Vector) -> tuple[Scalar, Vector]:
    """Integrating Neo-Hookean elasticity, a compliant base clamp, and an external `load` over `sample_volumes` gives the total energy and control-space force field."""
    displacement = sampling(state)                               # [...] Antivector-
    local = deformation(state, gradients)                        # [...] Antivector- <- Antivector-
    density, stress = constitutive(local, lame_lambda, lame_mu)
    density = density + clamp * displacement.scalar_norm_squared() / 2 - (load & displacement)
    energy = (density * sample_volumes).sum(axis=0)              # Scalar+
    elastic = (stress(gradients) * sample_volumes).sum(axis=0)   # Vector+[...]
    # Pulling the clamp restoring covector and external load back via the adjugate collects local forces onto the controls.
    applied = sampling.field(0).adjugate()(
        ((clamp * displacement.dual() - load) * sample_volumes).field()
    )                                                            # Vector+[...]
    return energy, elastic + applied
