"""Skeletal skinning deforms a triangulated surface over articulated bones by blending the bone transformations at each vertex according to its skinning weights. Mixing the destination points produced by each bone makes a twisted limb collapse toward the bone axis like a wrung-out towel, whereas mixing the rigid motions themselves keeps every vertex on a circle about the axis.

In 3D projective geometric algebra, a bone's rigid motion is modelled as an even versor called a motor, whose sandwich action on points forms the linear map `motor >> Point`. Linear blend skinning averages those linear maps before evaluating at each vertex, whereas motor skinning either normalises a linear blend of the motors or interpolates along their screw geodesic through `(tip / root).log()` and `.exp()`, so both `motor >> vertex` deformations remain rigid motions on every cross-section.
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


def cylinder(n_rings: int, n_around: int) -> tuple[Point, np.ndarray]:
    """Rotor turns around the x axis sweep a meridian from 0 to 1 into a unit-radius cylinder (`[n_rings * n_around] Point-`) with axial skinning weights."""
    weights, angles = np.meshgrid(
        np.linspace(0.0, 1.0, n_rings),
        np.linspace(0.0, 2 * np.pi, n_around, endpoint=False),
        indexing="ij",
    )
    # Turn the meridian line at y = 1, z = 0 around the x axis by each azimuth angle.
    meridian = mv.point(np.stack([weights, np.ones_like(weights), np.zeros_like(weights)], axis=-1))  # [n_rings, n_around] Point-
    turns = (mv.yz * (-angles / 2)).exp()                                                             # [n_rings, n_around] Motor+
    vertices = (turns >> meridian).reshape(-1)                                                        # [n_rings * n_around] Point-
    return vertices, weights.reshape(-1)


def blend_skin(vertices: Point, weights: np.ndarray, root: Motor, tip: Motor) -> tuple[Point, Point, Point]:
    """Normalised motor blending, motor slerp, and point-map blending each deform `vertices` (`[...] Point-`) between `root` and `tip`."""
    # Blend the bone motors linearly and project back onto the motor manifold by normalising.
    blended_motors = (root + weights * (tip - root)).normalized()                  # [...] Motor+
    motor_skin = blended_motors >> vertices                                        # [...] Point-

    # Slerp follows the screw geodesic through the relative motor's logarithm and exponential;
    # the normalised linear blend is a chord of the same arc and differs only in axial twist spacing.
    slerp_motors = ((tip / root).log() * weights).exp() * root                     # [...] Motor+
    slerp_skin = slerp_motors >> vertices                                          # [...] Point-

    # Linear blend skinning blends the two bones' point maps before evaluating at each vertex.
    blended_maps = (root >> Point) + weights * ((tip >> Point) - (root >> Point))  # [...] Point- <- Point-
    matrix_skin = blended_maps(vertices)                                           # [...] Point-
    return motor_skin, slerp_skin, matrix_skin
