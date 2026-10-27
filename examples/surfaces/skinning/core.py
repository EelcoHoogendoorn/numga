"""Skeletal skinning deforms a surface mesh over articulated bones by blending the bone transformations at each vertex according to its skinning weights. Mixing the destination points produced by each bone makes a twisted limb collapse toward the bone axis like a wrung-out towel, whereas mixing the rigid motions themselves keeps every vertex on a circle about the axis.

In 3D projective geometric algebra, a bone's rigid motion is a motor, and its action on points is the linear map `motor >> Point`. Linear blend skinning blends the two bones' point maps and applies the blended map; motor skinning blends the motors, normalises, and applies the sandwich, or follows the screw geodesic between them through the motor logarithm and exponential. Because the maps and the motors are extensors of the same kind, both blends take the same form, while only the motor blends remain rigid motions.

In the matrix notation of computer graphics the point map reads as a $4 \\times 4$ transformation matrix and linear blend skinning as the weighted matrix sum $((1 - w) M_0 + w M_1)\\mathbf p$. In dual quaternion notation a motor reads as a unit dual quaternion and normalised motor blending as dual quaternion linear blending.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext
from numga.algebras import PGA3D


ga = PGA3D
ctx = NumpyContext(ga)
mv = ctx.multivector
Point = ga.gatype.antivector()                      # Antivector-
Motor = ga.gatype.rotor()                           # Rotor+


# --- math -----------------------------------------------------------------------------
def blend_skin(vertices: Point, weights: np.ndarray, root: Motor, tip: Motor) -> tuple[Point, Point, Point]:
    """The vertices deformed by normalised motor blending, motor slerp, and point-map blending."""
    # Blend the bone motors linearly and normalise back to the motor manifold.
    blended_motors = (root + weights * (tip - root)).normalized()                  # [vertices] Motor+
    motor_skin = blended_motors >> vertices                                        # [vertices] Point-

    # Slerp follows the screw geodesic: the logarithm of the relative motor, scaled by the
    # weight and exponentiated. The normalised linear blend is a chord of the same arc, so both
    # are rigid about the axis and differ only in how their twist angles are spaced along x.
    slerp_motors = ((tip / root).log() * weights).exp() * root                     # [vertices] Motor+
    slerp_skin = slerp_motors >> vertices                                          # [vertices] Point-

    # Linear blend skinning blends the two bones' point maps and applies the blended map.
    blended_maps = (root >> Point) + weights * ((tip >> Point) - (root >> Point))  # [vertices] Point- <- Point-
    matrix_skin = blended_maps(vertices)                                           # [vertices] Point-
    return motor_skin, slerp_skin, matrix_skin


# --- plumbing -------------------------------------------------------------------------
def cylinder(n_rings: int, n_around: int) -> tuple[Point, np.ndarray]:
    """A unit-radius cylinder around the x axis from 0 to 1, with each vertex's weight equal to x."""
    x, angles = np.meshgrid(
        np.linspace(0.0, 1.0, n_rings),
        np.linspace(0.0, 2 * np.pi, n_around, endpoint=False),
        indexing="ij",
    )
    # Turn the generator line at y = 1, z = 0 around the x axis by each azimuth angle.
    ridge = mv.point(np.stack([x, np.ones_like(x), np.zeros_like(x)], axis=-1))    # [rings, around] Point-
    turns = (mv.yz * (-angles / 2)).exp()                                          # [rings, around] Motor+
    vertices = (turns >> ridge).reshape(-1)                                        # [vertices] Point-
    return vertices, x.reshape(-1)
