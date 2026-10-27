"""Stress in an elastic cube: the traction on each face, the principal frame, and Mohr's circle, in
the geometric algebra of three-dimensional space.

A strain is the change it makes to the metric: a map from points of the material to the planes that
read their stretch, `Strain = Plane <- Point`; the metric turns each reading into how far the point moves.
The stress is a map from planes to points, `Stress = Point <- Plane`: given a face's plane it returns the
traction, the force per area across that face, which splits into a part across the face, its pairing with
the face, and a shear along it. An isotropic material is where the metric enters the law, three times: it
turns a face's plane into the direction across it, `Plane.dual()`, takes the strain's trace, its change
of volume, `dilatation = strain(Plane.dual()).trace()`, and turns the strain's reading back into a direction,
so the stress is `lame * dilatation * Plane.dual() + 2 * shear_modulus * strain(Plane.dual()).dual()`, with
the first Lamé parameter and the shear modulus.

Seen from a turned frame, `rotor << stress(rotor >> Plane)`, the cube's faces feel other tractions. The
stress paired with a second plane is a form, `Plane & stress(Plane)`, and its eigenvectors against the
metric are the principal planes: in the principal frame the stress sends each face's plane to a multiple
of the direction across it and no face is sheared. Turning the frame in the shear plane, the normal and shear traction on one face trace a
circle, Mohr's, whose radius is the largest shear any face feels.

In tensor notation the stress reads as $\\sigma_{ij} = \\lambda\\,\\varepsilon_{kk}\\delta_{ij} +
2\\mu\\,\\varepsilon_{ij}$, and the traction on a face with unit normal $n$ as $t_i = \\sigma_{ij} n_j$.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext, concatenate
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
# Points, and how far they move, are antivectors with intrinsic orientation; a face is a plane.
Point = ga.gatype.antivector()
Plane = ga.gatype.vector()
Rotor = ga.gatype.rotor()
Strain = ga.gatype((Plane, Point))                         # Plane+ <- Point-
Stress = ga.gatype((Point, Plane))                         # Point- <- Plane+

axes = mv.basis()                                          # [3] Plane+
# The cube of unit side about the origin: its corners, and its faces' planes, facing out.
corners = mv.vector(np.stack(np.meshgrid(*[[-0.5, 0.5]] * 3, indexing="ij"), axis=-1).reshape(-1, 3)).dual()   # [8] Point-
faces = concatenate([axes, -axes])                         # [6] Plane+


# --- math -----------------------------------------------------------------------------
def cauchy_stress(strain: Strain, lame: float, shear_modulus: float) -> Stress:
    """The stress of an isotropic material under the strain."""
    # The strain's change of volume: its trace, read on the directions the metric gives the axes' planes.
    dilatation = strain(Plane.dual()).trace()                                  # [] Scalar+
    # The metric turns a face's plane into the direction across it, and the strain's reading back into one.
    return lame * dilatation * Plane.dual() + 2 * shear_modulus * strain(Plane.dual()).dual()   # [] Stress


def tractions(stress: Stress, faces: Plane) -> tuple[Point, Point]:
    """The traction on each face of the given unit plane, split into its part across the face and the
    shear along it."""
    total = stress(faces)                                                      # [...] Point-
    across = faces.dual() * (faces & total)                                    # [...] Point-
    return across, total - across


def principal_frame(stress: Stress) -> tuple[Scalar, Point, Rotor]:
    """The principal stresses, from least to greatest, the principal directions, and the turn taking
    the plane x onto the first principal plane: for a shear in the xy plane, the turn in that plane onto
    the principal frame."""
    # The stress paired with a second plane is a form; its eigenvectors against the metric are the
    # principal planes.
    values, planes = (Plane & stress(Plane)).eigh(Plane | Plane)              # [3] Scalar+, [3] Plane+
    # The metric turns each principal plane into the direction across it.
    return values, planes.dual(), (1 + planes[0] * ~axes[0]).normalized()


def mohr_circle(values: Scalar) -> tuple[Scalar, Scalar]:
    """The centre of Mohr's circle in the shear plane, and its radius, the largest shear: from the
    least and the greatest principal stress."""
    return (values[0] + values[2]) / 2, (values[2] - values[0]).abs() / 2


def views(strain: Strain, stress: Stress, directions: Point, rotors: Rotor
          ) -> tuple[Point, Point, Point, Point, Point]:
    """The cube seen from each frame `[...]`: the strain and the stress as each frame sees them, the cube's
    deformed corners `[..., 8]` and face centres `[..., 6]`, the normal and shear traction on each face
    `[..., 6]`, and the principal directions `[..., 3]`."""
    local_strain = rotors << strain(rotors >> Point)                           # [...] Strain
    local_stress = rotors << stress(rotors >> Plane)                           # [...] Stress
    normal, shear = tractions(local_stress[..., None], faces)                 # [..., 6] Point-
    # The metric turns each point's strain reading into how far it moves.
    return (corners + local_strain[..., None](corners).dual(),                  # [..., 8] Point-
            0.5 * (faces.dual() + local_strain[..., None](faces.dual()).dual()),   # [..., 6] Point-
            normal, shear, rotors[..., None] << directions)                    # [..., 3] Point-
