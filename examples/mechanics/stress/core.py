"""Stress in an elastic cube: the traction on each face, the principal frame, and Mohr's circle, in
the geometric algebra of three-dimensional space.

A strain is a map on vectors, `Strain = Vector <- Vector`: how far each point of the material moves,
for where it sits. In an isotropic material the stress is a map of the same kind,
`lame * strain.trace() + 2 * shear_modulus * strain`, with the first Lamé parameter and the shear
modulus. Given a face's normal it returns the traction, the force per area across that face, which
splits into a part along the normal and a shear along the face.

Seen from a turned frame, `rotor << stress(rotor >> Vector)`, the cube's faces feel other tractions.
In the principal frame the stress sends each face's normal to a multiple of itself and no face is
sheared. Turning the frame in the shear plane, the normal and shear traction on one face trace a
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
Vector = ga.gatype.vector()
Rotor = ga.gatype.rotor()
Strain = ga.gatype((Vector, Vector))                       # Vector <- Vector
Stress = ga.gatype((Vector, Vector))                       # Vector <- Vector

axes = mv.basis()                                          # [3] Vector
# The cube of unit side about the origin: its corners, and its faces' outward normals.
corners = mv.vector(np.stack(np.meshgrid(*[[-0.5, 0.5]] * 3, indexing="ij"), axis=-1).reshape(-1, 3))   # [8] Vector
normals = concatenate([axes, -axes])                       # [6] Vector


# --- math -----------------------------------------------------------------------------
def cauchy_stress(strain: Strain, lame: float, shear_modulus: float) -> Stress:
    """The stress of an isotropic material under the strain."""
    identity = mv.rotor() >> Vector                                            # [] Vector <- Vector
    return lame * strain.trace() * identity + 2 * shear_modulus * strain     # [] Stress


def tractions(stress: Stress, faces: Vector) -> tuple[Vector, Vector]:
    """The traction on each face of the given normal, split into its part along the normal and the
    shear along the face."""
    total = stress(faces)                                                      # [...] Vector
    along = faces * (faces | total)                                            # [...] Vector
    return along, total - along


def principal_frame(stress: Stress) -> tuple[Scalar, Vector, Rotor]:
    """The principal stresses, from least to greatest, the principal directions, and the turn taking
    x onto the first of them: for a shear in the xy plane, the turn in that plane onto the
    principal frame."""
    values, directions = stress.eigh()                                         # [3] Scalar, [3] Vector
    return values, directions, (1 + directions[0] * axes[0]).normalized()


def mohr_circle(values: Scalar) -> tuple[Scalar, Scalar]:
    """The centre of Mohr's circle in the shear plane, and its radius, the largest shear: from the
    least and the greatest principal stress."""
    return (values[0] + values[2]) / 2, (values[2] - values[0]).abs() / 2


class Views:
    """The cube seen from each of a batch of frames `[frames]`: its deformed corners `[frames, 8]`, the
    centres of its deformed faces `[frames, 6]`, the normal and shear traction on each face
    `[frames, 6]`, and the principal directions `[frames, 3]`."""

    def __init__(self, strain: Strain, stress: Stress, directions: Vector, rotors: Rotor) -> None:
        local_strain = rotors << strain(rotors >> Vector)                      # [frames] Strain
        local_stress = rotors << stress(rotors >> Vector)                      # [frames] Stress
        self.corners = corners + local_strain[:, None](corners)               # [frames, 8] Vector
        self.centres = 0.5 * (normals + local_strain[:, None](normals))       # [frames, 6] Vector
        self.normal, self.shear = tractions(local_stress[:, None], normals)  # [frames, 6] Vector
        self.principal = rotors[:, None] << directions                        # [frames, 3] Vector
