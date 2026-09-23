"""Which responses survive averaging over an object's rotation symmetries?

Heat conduction: a positive conductivity, averaged over rotation groups, and its heat-flow ellipsoids.
Flywheel: a flywheel with three arms, assembled by adding their rotated inertia maps.
Crystal lattice: the same cubic bond orbits give isotropic conduction and anisotropic elasticity.

Crystal-symmetry motivation: https://dictionary.iucr.org/Neumann%27s_principle
"""

from __future__ import annotations

import numpy as np

from numga import Algebra, Extensor, NumpyContext
from numga.algebras import PGA3D

# Conductivity acts on Euclidean driving-field and heat-flow vectors.
ga = Algebra("x+y+z+")
ctx = NumpyContext(ga)
mv = ctx.multivector
Scalar = ga.gatype.scalar()
Vector = ga.gatype.vector()
Rotor = ga.gatype.rotor()
Plane = ga.gatype.bivector()

# Mechanics uses projective points and a momentum response to rigid-body motion.
pga_mv = NumpyContext(PGA3D).multivector
Point = PGA3D.gatype.antivector()
Bivector = PGA3D.gatype.bivector()
AntiBivector = PGA3D.gatype.antibivector()


# --- construction ---------------------------------------------------------------------
def turns(plane: Plane, order: int) -> Rotor:
    """The cyclic group of `order` equal turns in a unit plane, the identity first."""
    # A rotor turns by twice its angle: exp(plane * pi k / order) turns by 2 pi k / order.
    return (plane * (np.pi * np.arange(order) / order)).exp()


def cube_rotations() -> Rotor:
    """The 24 rotations of a cube: each of the six faces turned to the top, then four quarter turns about z."""
    tilt = np.pi / 4 * np.array([0, 1, 2, 3, 0, 0])
    roll = np.pi / 4 * np.array([0, 0, 0, 0, 1, -1])
    faces = (mv.yz * tilt + mv.zx * roll).exp()
    return (faces[:, None] * turns(mv.xy, 4)[None, :]).reshape(-1)


def directions() -> Vector:
    """A sphere of unit driving fields, sampled for drawing."""
    longitude = np.linspace(0, 2 * np.pi, 65)[None, :]
    latitude = np.linspace(-np.pi / 2, np.pi / 2, 33)[:, None]
    return (mv.x * (np.cos(latitude) * np.cos(longitude))
            + mv.y * (np.cos(latitude) * np.sin(longitude))
            + mv.z * np.broadcast_to(np.sin(latitude), (33, 65)))


def arm_samples() -> Point:
    """Equal-mass samples of a thin rectangular flywheel arm attached to a massless hub."""
    translations = (pga_mv.xw * np.linspace(0.25, 2.2, 24)[:, None]
                    + pga_mv.yw * np.linspace(-0.14, 0.14, 5)[None, :]) / 2
    return translations.exp() >> pga_mv.zyx


def lattice_samples():
    """PGA sites of a simple-cubic lattice, with the central site's two neighbour shells."""
    i, j, k = np.indices((3, 3, 3)) - 1
    sites = pga_mv.zyx + pga_mv.yzw * i + pga_mv.zxw * j + pga_mv.xyw * k
    shell = i * i + j * j + k * k
    return sites.reshape(-1), sites[shell == 1], sites[shell == 2]


# --- math -----------------------------------------------------------------------------
def conductivities(axes: Vector, gains: Scalar, groups: list[Rotor]) -> Extensor:
    """Which heat-conduction responses are compatible with a crystal's symmetry?

    Conductivity maps the negative temperature gradient to heat flow. In an anisotropic
    material these vectors need not be parallel. Start with a positive conductivity,
    rotate the whole response through each symmetry, and average. This Reynolds
    projection gives the closest invariant response in the Frobenius norm, preserving
    positive definiteness. Symmetry constrains the response, not its numerical gains.

    Returns the measured conductivity followed by its average over each group.
    """
    # Each principal axis measures one component of the driving field and contributes
    # heat flow along that axis. Positive gains make this a passive conductivity.
    conductivity = (axes * (axes | Vector) * gains).sum(axis=0)
    responses = [conductivity]
    for group in groups:
        # Pull the input into each rotated frame, apply K, and rotate the output back.
        # The mean is unchanged by any rotation in the group: those rotations merely
        # permute the terms. This projects a measured response onto the allowed ones.
        invariant = (group >> Vector)(conductivity(group << Vector)).mean(axis=0)
        responses.append(invariant)
    return Extensor.stack(responses)


def flywheel_inertia(arm: Point, rotations: Rotor) -> tuple[Extensor, Extensor]:
    """Assemble a flywheel by adding rotated copies of one arm's inertia.

    Each arm has unit mass. Threefold symmetry puts the center of mass at the hub
    and makes all transverse moments equal, even though the shape has only three arms.
    Rotation about z still has a different moment: for this thin planar object it is
    twice the transverse moment.

    Mass locations are PGA points (antivectors). Inertia maps bivector rigid-body
    velocities to antibivector momenta, including translation as well as rotation.
    Returns the inertia of one arm and of the whole flywheel.
    """
    # The commutator gives each point's velocity under the open rigid-motion slot.
    # Join point and velocity to get momentum, then average for a unit-mass arm.
    arm_inertia = (arm & arm.commutator(Bivector)).mean()

    # Rotate that entire response and add the three arms. Unlike conductivity's mean,
    # this sum assembles actual masses: the finished flywheel has mass three.
    inertia = (rotations >> AntiBivector)(arm_inertia(rotations << Bivector)).sum(axis=0)
    return arm_inertia, inertia


def lattice_responses(seeds: Vector, cube: Rotor) -> tuple[Extensor, Extensor]:
    """Average microscopic bond responses over a crystal's cubic point group.

    A simple-cubic lattice has axial and face-diagonal bond families. Starting with
    x and normalized(x+y), the 24 cube rotations generate every direction in each
    family. Every member has the same weight, so another cube rotation just permutes
    the terms in the average. This is the microscopic meaning of the group projection.

    Give the two families equal total response weights. For the elastic model, take
    unstressed central springs with lattice spacing 1, axial stiffness 1/3, and
    face-diagonal stiffness 1/12. Bond counts and squared lengths give weight 1 to
    each family. Diagonal springs provide shear stiffness. This particular model has
    C12 = C44; a general cubic crystal need not have that extra relation.

    A unit bond n measures a thermal gradient through n·g. For imposed strain
    strain(v) = s*d*(d·v), its fractional extension is s*(n·d)^2. Squaring to get
    spring energy gives four factors of n·d, hence a scalar form with four vector
    slots. Its value C(d,d,d,d) is the stiffness, with energy density s^2*C/2.
    It holds transverse strain fixed; it is not the relaxed Young's modulus.

    Cubic elasticity background: https://www.ctcms.nist.gov/oof/oof1/Manual/node152.html
    Returns the conductivity and the elasticity.
    """
    # Batch axes are [24 rotations, 2 bond families]. These are free bond directions;
    # the actual lattice sites are projective points.
    bonds = cube[:, None] >> seeds[None, :]
    projection = bonds | Vector                         # Scalar <- Vector

    # Average over the group, then add the two families. No direction is privileged
    # within either orbit. Conductivity measures the gradient and returns flow along n.
    conductivity = (bonds * projection).mean(axis=0).sum(axis=0)

    # Four independent slots encode the spring stiffness. The same group average
    # leaves cubic structure here, although it made the vector response isotropic.
    elasticity = (projection * projection * projection * projection).mean(axis=0).sum(axis=0)
    return conductivity, elasticity
