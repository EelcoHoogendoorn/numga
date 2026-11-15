"""Rigid-body stiffness and inertia map twists to forques and determine the vibration modes of a suspended plate.

Joining each spring's anchor to its attachment gives its line of action, pairing that line with `Twist` reads the spring extension, and multiplying the line back by that linear form scaled by the spring constant builds a Hookean dyad from twists to forques. Summing those dyads over the springs gives the suspension stiffness, while summing `(masses * mass_points).regressive(mass_points.commutator(Twist))` over the lumped mass points gives the rigid-body inertia without choosing a reference point or shifting axes.

Pairing both maps with a second `Twist` turns stiffness and inertia into symmetric energy forms on twists whose generalized eigenproblem `potential_form.eigh(kinetic_form)` yields the natural frequencies and mass-normalized mode twists.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import NumpyContext
from numga.algebras import PGA2D


ga = PGA2D
mv = NumpyContext(ga).multivector

Scalar = ga.gatype.scalar()                          # Scalar+
Point = ga.gatype.antivector()                       # Antivector-
Twist = ga.gatype.bivector()                         # Bivector+
Forque = ga.gatype.antibivector()                    # Antibivector-
Stiffness = ga.gatype((Forque, Twist))               # Forque- <- Twist+
Inertia = ga.gatype((Forque, Twist))                 # Forque- <- Twist+


def quadrature_points(corners: Point) -> Point:
    """Two-point Gauss quadrature places four equal-weight mass nodes (`[4] Point-`) inside a rectangular plate with `corners` (`[4] Point-`)."""
    # Two-point Gauss quadrature integrates quadratic mass moments on the rectangle with four equal weights.
    centre = corners.sum(axis=0) / 4                                             # [] Point-
    offsets = corners - centre                                                   # [4] Point-
    return centre + offsets / np.sqrt(3)


def body_inertia(mass_points: Point, masses: np.ndarray) -> Inertia:
    """Momentum lines across `mass_points` (`[...] Point-`) weighted by `masses` sum into the rigid-body inertia (`[] Forque- <- Twist+`)."""
    # Joining each mass-weighted point with its commutator velocity builds its momentum map without a reference point.
    velocities = mass_points.commutator(Twist)                                   # [...] Point- <- Twist+
    point_momenta = (masses * mass_points).regressive(velocities)                # [...] Forque- <- Twist+
    return point_momenta.sum(axis=0)


def spring_stiffness(anchors: Point, attachments: Point, spring_constants: np.ndarray) -> Stiffness:
    """Hookean dyads of springs joining `anchors` to `attachments` (`[...] Point-`) with `spring_constants` sum into the stiffness map (`[] Forque- <- Twist+`)."""
    # Pairing a twist with each unit spring line reads its first-order extension, and multiplying back pulls along the same line.
    lines = (anchors & attachments).normalized()                                 # [...] Forque-
    extensions = Twist & lines                                                   # [...] Scalar+ <- Twist+
    dyads = spring_constants * lines * extensions                                # [...] Forque- <- Twist+
    return dyads.sum(axis=0)


def normal_modes(stiffness: Stiffness, inertia: Inertia) -> tuple[Scalar, Twist]:
    """Generalized eigenanalysis of the potential and kinetic energy forms yields the natural frequencies (`[3] Scalar+`, in hertz) and mass-normalized mode twists (`[3] Twist+`)."""
    # Pairing stiffness and inertia with a second twist turns both maps into symmetric bilinear energy forms.
    potential_form = Twist & stiffness                                           # [] Scalar+ <- (Twist+, Twist+)
    kinetic_form = Twist & inertia                                               # [] Scalar+ <- (Twist+, Twist+)
    eigenvalues, mode_twists = potential_form.eigh(kinetic_form)                 # [3] Scalar+, Twist+
    frequencies = eigenvalues.clip(0, np.inf).square_root() / (2 * np.pi)        # [3] Scalar+
    return frequencies, mode_twists


def mode_oscillation(frequencies: Scalar, mode_twists: Twist, n_frames: int) -> Iterator[Twist]:
    """Yields `n_frames` harmonic twist states (`[3] Twist+`) over two periods of the lowest nonzero frequency."""
    duration = 2 / frequencies[frequencies > 0].min()                            # [] Scalar+
    for fraction in np.linspace(0.0, 1.0, n_frames):
        phases = (2 * np.pi * frequencies * (fraction * duration)).cos()         # [3] Scalar+
        yield phases * mode_twists
