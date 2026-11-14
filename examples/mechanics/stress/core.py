"""Stress in an elastic cube relates the traction on each face, the principal frame, and Mohr's
circle in the geometric algebra of three-dimensional space.

A strain is modelled as a map from material points to the planes that read their stretch,
`Plane <- Point`, and the metric turns each reading into how far the point moves. The Cauchy stress
maps each face's plane to the force per area across that face, `Point <- Plane`, which splits into a
normal part along `faces.dual()` and a shear part in the face. In an isotropic material, the metric
turns a face's plane into the direction across it (`Plane.dual()`), reads the dilatation from the
strain's trace (`strain(Plane.dual()).trace()`), and turns the strain's plane reading back into a
direction, giving `lame * dilatation * Plane.dual() + 2 * shear_modulus * strain(Plane.dual()).dual()`.

Seen from a rotated frame, the stress transforms as `rotor << stress(rotor >> Plane)`. Pairing the
stress with a second plane builds the symmetric form `Plane & stress(Plane)`, whose generalized
eigenvectors against `Plane | Plane` are the principal planes where shear vanishes. As the frame
turns through the shear plane, the normal and shear tractions on a face trace Mohr's circle, whose
radius equals the greatest shear traction on any face.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import NumpyContext, concatenate
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                # Scalar+
# Points and displacements are intrinsic antivectors; cut faces are extrinsic planes.
Point = ga.gatype.antivector()                             # Antivector-
Plane = ga.gatype.vector()                                 # Vector+
Bivector = ga.gatype.bivector()                            # Bivector+
Rotor = ga.gatype.rotor()                                  # Rotor+
Strain = ga.gatype((Plane, Point))                         # Plane+ <- Point-
Stress = ga.gatype((Point, Plane))                         # Point- <- Plane+

axes = mv.basis()                                          # [3] Plane+
# Unit cube centred on the origin: eight corners and six outward-facing planes.
corners = mv.vector(np.stack(np.meshgrid(*[[-0.5, 0.5]] * 3, indexing="ij"), axis=-1).reshape(-1, 3)).dual()   # [8] Point-
faces = concatenate([axes, -axes])                         # [6] Plane+


# --- math -----------------------------------------------------------------------------
def cauchy_stress(strain: Strain, lame: float, shear_modulus: float) -> Stress:
    """Isotropic Hooke's law maps `strain` to the Cauchy stress (`[] Point- <- Plane+`) with first
    Lamé parameter `lame` and shear modulus `shear_modulus`."""
    # Trace of the strain read along the metric duals of the basis planes gives the volume dilatation.
    dilatation = strain(Plane.dual()).trace()                                  # [] Scalar+
    # Dualizing `Plane` gives the unit normal direction across a face, and dualizing `strain(...)` turns its plane reading back into a traction direction.
    return lame * dilatation * Plane.dual() + 2 * shear_modulus * strain(Plane.dual()).dual()   # [] Point- <- Plane+


def face_tractions(stress: Stress, cut_faces: Plane) -> tuple[Point, Point]:
    """Each unit plane in `cut_faces` (`[...] Plane+`) receives a traction under `stress` that splits
    into its normal component (`[...] Point-`) across the face and its shear component (`[...] Point-`)
    along the face."""
    total = stress(cut_faces)                                                  # [...] Point-
    normal = cut_faces.dual() * (cut_faces & total)                            # [...] Point-
    return normal, total - normal


def principal_frame(stress: Stress) -> tuple[Scalar, Point, Rotor]:
    """Solving the symmetric eigenproblem of `Plane & stress(Plane)` against `Plane | Plane` gives
    the ordered principal stresses (`[3] Scalar+`), the principal directions (`[3] Point-`), and the
    rotor (`[] Rotor+`) carrying the first basis plane onto the first principal plane."""
    # Pairing `stress(Plane)` with `Plane` forms a symmetric bilinear form whose eigenvectors against `Plane | Plane` are the principal planes.
    stresses, planes = (Plane & stress(Plane)).eigh(Plane | Plane)             # [3] Scalar+, [3] Plane+
    return stresses, planes.dual(), (1 + planes[0] * ~axes[0]).normalized()


def mohr_circle(stresses: Scalar) -> tuple[Scalar, Scalar]:
    """The least and greatest of the ordered principal `stresses` (`[3] Scalar+`) set the normal-stress
    centre (`[] Scalar+`) and maximum-shear radius (`[] Scalar+`) of the outer Mohr circle."""
    return (stresses[0] + stresses[2]) / 2, (stresses[2] - stresses[0]).abs() / 2


def frame_views(strain: Strain, stress: Stress, directions: Point, rotors: Rotor
                ) -> tuple[Point, Point, Point, Point, Point]:
    """Transforming `strain`, `stress`, and the principal `directions` (`[3] Point-`) into each frame
    of `rotors` (`[...] Rotor+`) gives the deformed cube corners (`[..., 8] Point-`), face centres
    (`[..., 6] Point-`), normal and shear face tractions (`[..., 6] Point-`), and local principal
    directions (`[..., 3] Point-`)."""
    local_strain = rotors << strain(rotors >> Point)                           # [...] Plane+ <- Point-
    local_stress = rotors << stress(rotors >> Plane)                           # [...] Point- <- Plane+
    normal, shear = face_tractions(local_stress[..., None], faces)             # [..., 6] Point-, [..., 6] Point-
    # Dualizing each point's strain plane gives its displacement in the local frame.
    return (corners + local_strain[..., None](corners).dual(),                 # [..., 8] Point-
            0.5 * (faces.dual() + local_strain[..., None](faces.dual()).dual()),   # [..., 6] Point-
            normal, shear, rotors[..., None] << directions)                    # [..., 3] Point-


def turning_views(strain: Strain, stress: Stress, directions: Point, principal: Rotor, plane: Bivector,
                  n_frames: int) -> Iterator[tuple[Point, Point, Point, Point, Point]]:
    """Yields `frame_views` across `n_frames` evenly spaced angles over half a turn in `plane`
    (`[] Bivector+`) starting from `principal`."""
    for angle in np.linspace(0.0, np.pi, n_frames, endpoint=False):
        rotor = (plane * (-angle / 2)).exp() * principal                       # [] Rotor+
        yield frame_views(strain, stress, directions, rotor)
