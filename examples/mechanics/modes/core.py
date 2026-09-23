"""Planar rigid-body stiffness, assembled from spring lines with an open twist.

A spring joins a fixed anchor to a point on the body. Its normalized PGA line
is both its line of action and a measurement: pairing it with a small rigid
motion gives the spring's extension. Leave that motion open, multiply by the
same line and the spring constant, and sum. The result is a stiffness extensor
mapping body displacement to the opposing wrench (force and torque).

Pair the output with another open motion to obtain the bilinear energy form.
The inertia extensor uses the same input and output spaces; their generalized
eigenvectors are the body's normal modes. No stiffness or mass matrix entries
are written by hand: the forms go directly to the library eigensolver.

Two vertical springs allow a sideways slide, a bounce and a rocking motion.
Adding an off-centre angled spring removes the free slide and couples these
motions. The plots show each mode's displacement and which springs stretch.

This is a small-motion model about an unstressed equilibrium, with no gravity
or damping. Springs carry both tension and compression. The sideways mode is
free to first order, not an exact finite mechanism: sideways motion changes
the vertical springs' lengths at second order. Displayed displacements are
enlarged linear mode shapes. A zero-frequency mode released at rest remains
displaced; the other modes oscillate at their computed natural frequencies.

Run from the repository root with python -m examples.mechanics.modes.scenarios.
Add --animate to also save plots/modes.gif.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from numga import NumpyContext
from numga.algebras import PGA2D


ctx = NumpyContext(PGA2D)
mv = ctx.multivector

Scalar = PGA2D.gatype.scalar()
Point = PGA2D.gatype.antivector()
Twist = PGA2D.gatype.bivector()
Wrench = PGA2D.gatype.antibivector()
SpringExtension = PGA2D.gatype((Scalar, Twist))
Stiffness = PGA2D.gatype((Wrench, Twist))
Inertia = PGA2D.gatype((Wrench, Twist))


def point(coords: np.ndarray) -> Point:
    """Finite points at (..., 2) coordinates: the dual of the homogeneous vector."""
    return (mv("x y", coords) + mv.w).dual()


@dataclass(frozen=True)
class Suspension:
    """Geometry and masses prepared for one planar suspension."""

    body: Point                     # [4] Point
    attachments: Point              # [n_springs] Point
    anchors: Point                  # [n_springs] Point
    spring_constants: np.ndarray    # [n_springs] float
    mass_points: Point              # [4] Point
    masses: np.ndarray              # [4] float


@dataclass(frozen=True)
class ModeCase:
    """The mode shapes of one suspension: displacements of body and springs per mode."""

    body: Point                     # [4] Point
    attachments: Point              # [n_springs] Point
    anchors: Point                  # [n_springs] Point
    frequencies: Scalar             # [3] Scalar, in Hz
    body_offsets: Point             # [3, 4] Point
    attachment_offsets: Point       # [3, n_springs] Point
    extensions: Scalar              # [3, n_springs] Scalar


def suspension(springs: int) -> Suspension:
    """A uniform 2-by-1 plate of mass 1, on the first `springs` springs of stiffness 6.

    The first two springs hang vertically; the third is angled and off-centre.
    """
    body_xy = np.array([[0.0, 0.5], [2.0, 0.5], [2.0, 1.5], [0.0, 1.5]])          # [4, 2] float (center at (1, 1))
    attachments_xy = np.array([[0.2, 1.5], [1.8, 1.5], [2.0, 1.0]])              # [3, 2] float
    anchors_xy = np.array([[0.2, 2.55], [1.8, 2.55], [2.9, 1.85]])               # [3, 2] float
    body = point(body_xy)                                                         # [4] Point
    attachments = point(attachments_xy[:springs])                                 # [n_springs] Point
    anchors = point(anchors_xy[:springs])                                         # [n_springs] Point
    spring_constants = np.full(springs, 6.0)                                      # [n_springs] float

    # Tensor-product 2-point Gauss quadrature on the uniform plate:
    center = body.sum(axis=0) * 0.25
    mass_points = center + (body - center) / np.sqrt(3)                           # [4] Point
    masses = np.full(4, 0.25)                                                     # [4] float
    return Suspension(body, attachments, anchors, spring_constants, mass_points, masses)


def mode_case(
    body: Point,
    attachments: Point,
    anchors: Point,
    values: Scalar,
    body_offsets: Point,
    attachment_offsets: Point,
    extensions: Scalar,
) -> ModeCase:
    """Collect mode geometry, with natural frequencies f = sqrt(lambda) / (2 pi) in Hz."""
    frequencies = values.clip(0, np.inf).square_root() / (2 * np.pi)              # [3] Scalar
    return ModeCase(body, attachments, anchors, frequencies, body_offsets, attachment_offsets, extensions)


def normal_modes(system: Suspension) -> ModeCase:
    """Stiffness and inertia of one suspension, and its three modes."""
    # 1. Spring lines of action in PGA (joining anchor to attachment):
    lines: Wrench = (system.anchors & system.attachments).normalized()           # [n_springs] Wrench

    # 2. Pairing an open twist with the spring line measures linear stretch (Twist -> Scalar):
    extension: SpringExtension = Twist & lines                                   # [n_springs] Scalar <- Twist

    # 3. Hooke's law: line of action scaled by extension and spring constant (Wrench <- Twist):
    spring_stiffness: Stiffness = lines * extension * system.spring_constants    # [n_springs] Wrench <- Twist
    stiffness: Stiffness = spring_stiffness.sum(axis=0)                          # [] Wrench <- Twist

    # 4. Direction of motion (velocity) of each point under an open twist:
    velocities: Point = system.mass_points.commutator(Twist)                     # [4] Point <- Twist
    point_momenta: Inertia = (system.mass_points & velocities) * system.masses   # [4] Wrench <- Twist
    inertia: Inertia = point_momenta.sum(axis=0)                                 # [] Wrench <- Twist

    # 5. Bilinear energy forms (Scalar <- Twist, Twist):
    pe_form = Twist & stiffness                                                  # [] Scalar <- (Twist, Twist)
    ke_form = Twist & inertia                                                    # [] Scalar <- (Twist, Twist)

    # 6. Solve generalized symmetric eigenvalue problem directly on bilinear energy forms:
    values, modes = pe_form.eigh(ke_form)                                        # values: [3] Scalar, modes: [3] Twist

    # 7. Evaluate physical displacements (Lie bracket) and spring extensions:
    body_offsets: Point = system.body[None, :].commutator(modes[:, None])        # [3, 4] Point
    attachment_offsets: Point = system.attachments[None, :].commutator(modes[:, None])  # [3, n_springs] Point
    extensions: Scalar = extension(modes[:, None])                               # [3, n_springs] Scalar

    return mode_case(
        body=system.body,
        attachments=system.attachments,
        anchors=system.anchors,
        values=values,
        body_offsets=body_offsets,
        attachment_offsets=attachment_offsets,
        extensions=extensions,
    )
