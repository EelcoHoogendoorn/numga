"""Vibration modes of a suspended plate in PGA2D: stiffness and inertia as maps from twists to forques.

A spring is the line from its anchor to its attachment; paired with a twist, `Twist & line`, it reads
how far the motion stretches the spring, and the spring's stiffness is the dyad
`line * (Twist & line) * k`, a map from twists to forques. The plate's stiffness is the sum over its
springs, and its inertia the sum over its mass points, each joined with its own motion under a twist,
`points & points.commutator(Twist)`. Paired back with a twist, both maps are energies, symmetric forms
on twists, and the natural modes are the twists where the two energies are proportional: one
eigenproblem of one form against the other.

Two vertical springs let the plate slide sideways freely, bounce and rock; an angled spring removes the
free slide and mixes the three. The model is linear about an unstressed equilibrium, so the slide is
free to first order only.
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
Forque = PGA2D.gatype.antibivector()
SpringExtension = PGA2D.gatype((Scalar, Twist))
Stiffness = PGA2D.gatype((Forque, Twist))
Inertia = PGA2D.gatype((Forque, Twist))


@dataclass(frozen=True)
class Suspension:
    """Geometry and masses prepared for one planar suspension."""

    body: Point                     # [corners] Point
    attachments: Point              # [n_springs] Point
    anchors: Point                  # [n_springs] Point
    spring_constants: np.ndarray    # [n_springs] N/m
    mass_points: Point              # [mass_points] Point
    masses: np.ndarray              # [mass_points] kg


@dataclass(frozen=True)
class ModeCase:
    """The mode shapes of one suspension: displacements of body and springs per mode."""

    body: Point                     # [corners] Point
    attachments: Point              # [n_springs] Point
    anchors: Point                  # [n_springs] Point
    frequencies: Scalar             # [modes] Scalar, in Hz
    body_offsets: Point             # [modes, corners] Point
    attachment_offsets: Point       # [modes, n_springs] Point
    extensions: Scalar              # [modes, n_springs] Scalar


def suspension(springs: int) -> Suspension:
    """A uniform 2-by-1 plate of mass 1, on the first `springs` springs of stiffness 6.

    The first two springs hang vertically; the third is angled and off-centre.
    """
    # The plate's corners, centred at (1, 1).
    body_xy = np.array([[0.0, 0.5], [2.0, 0.5], [2.0, 1.5], [0.0, 1.5]])          # [corners, 2] m
    attachments_xy = np.array([[0.2, 1.5], [1.8, 1.5], [2.0, 1.0]])              # [springs, 2] m
    anchors_xy = np.array([[0.2, 2.55], [1.8, 2.55], [2.9, 1.85]])               # [springs, 2] m
    body = mv.point(body_xy)                                                      # [corners] Point
    attachments = mv.point(attachments_xy[:springs])                              # [n_springs] Point
    anchors = mv.point(anchors_xy[:springs])                                      # [n_springs] Point
    spring_constants = np.full(springs, 6.0)                                      # [n_springs] N/m

    # Tensor-product 2-point Gauss quadrature on the uniform plate:
    center = body.sum(axis=0) * 0.25
    mass_points = center + (body - center) / np.sqrt(3)                           # [mass_points] Point
    masses = np.full(4, 0.25)                                                     # [mass_points] kg
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
    """Collect mode geometry, with natural frequencies in Hz: the square root of each eigenvalue
    over `2 * np.pi`."""
    frequencies = values.clip(0, np.inf).square_root() / (2 * np.pi)              # [modes] Scalar
    return ModeCase(body, attachments, anchors, frequencies, body_offsets, attachment_offsets, extensions)


def normal_modes(system: Suspension) -> ModeCase:
    """Stiffness and inertia of one suspension, and its three modes."""
    # Each spring's line, from its anchor to its attachment.
    lines: Forque = (system.anchors & system.attachments).normalized()           # [n_springs] Forque

    # How far a twist stretches each spring: the twist paired with the spring's line.
    extension: SpringExtension = Twist & lines                                   # [n_springs] Scalar <- Twist

    # Each spring pulls back along its line by its stretch times its constant, Hooke's law.
    spring_stiffness: Stiffness = lines * extension * system.spring_constants    # [n_springs] Forque <- Twist
    stiffness: Stiffness = spring_stiffness.sum(axis=0)                          # [] Forque <- Twist

    # Each mass point's velocity as a map of the twist, and its momentum, the line through it along that
    # velocity; the inertia is their sum.
    velocities: Point = system.mass_points.commutator(Twist)                     # [mass_points] Point <- Twist
    point_momenta: Inertia = (system.mass_points & velocities) * system.masses   # [mass_points] Forque <- Twist
    inertia: Inertia = point_momenta.sum(axis=0)                                 # [] Forque <- Twist

    # Each map paired with a second twist is an energy, a form on two twists.
    pe_form = Twist & stiffness                                                  # [] Scalar <- (Twist, Twist)
    ke_form = Twist & inertia                                                    # [] Scalar <- (Twist, Twist)

    # The natural modes: the twists where the two energies are proportional.
    values, modes = pe_form.eigh(ke_form)                                        # values: [modes] Scalar, modes: [modes] Twist

    # How each mode moves the corners and the attachments, and stretches the springs.
    body_offsets: Point = system.body[None, :].commutator(modes[:, None])        # [modes, corners] Point
    attachment_offsets: Point = system.attachments[None, :].commutator(modes[:, None])  # [modes, n_springs] Point
    extensions: Scalar = extension(modes[:, None])                               # [modes, n_springs] Scalar

    return mode_case(
        body=system.body,
        attachments=system.attachments,
        anchors=system.anchors,
        values=values,
        body_offsets=body_offsets,
        attachment_offsets=attachment_offsets,
        extensions=extensions,
    )
