"""Planar rigid-body stiffness, assembled from spring lines with an open twist.

A spring joins a fixed anchor to a point on the body. Its normalized PGA line
is both its line of action and a measurement: pairing it with a small rigid
motion gives the spring's extension. Leave that motion open, multiply by the
same line and the spring constant, and sum. The result is a stiffness extensor
mapping body displacement to the opposing wrench (force and torque).

Pair the output with another open motion to obtain the bilinear energy form.
The inertia extensor uses the same input and output spaces; their generalized
eigenvectors are the body's normal modes. No stiffness or mass matrix entries
are written by hand: coefficients are exposed only for the eigenproblem.

Two vertical springs allow a sideways slide, a bounce and a rocking motion.
Adding an off-centre angled spring removes the free slide and couples these
motions. The plots show each mode's displacement and which springs stretch.

This is a small-motion model about an unstressed equilibrium, with no gravity
or damping. Springs carry both tension and compression. The sideways mode is
free to first order, not an exact finite mechanism: sideways motion changes
the vertical springs' lengths at second order. Displayed displacements are
enlarged linear mode shapes. A zero-frequency mode released at rest remains
displaced; the other modes oscillate at their computed natural frequencies.

Run from rewrite/ with PYTHONPATH=src:. python -m examples.mechanics.stiffness.
Add --animate to also save plots/stiffness.gif.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

from examples import PLOT_DIR
from numga import NumpyContext
from numga.algebras import PGA2D


ctx = NumpyContext(PGA2D)
mv = ctx.multivector

Scalar = PGA2D.gatype.scalar()
Point = PGA2D.gatype.antivector()
Twist = PGA2D.gatype.bivector()
Wrench = PGA2D.gatype.vector()
SpringExtension = PGA2D.gatype((Scalar, Twist))
Stiffness = PGA2D.gatype((Wrench, Twist))
Inertia = PGA2D.gatype((Wrench, Twist))
EnergyForm = PGA2D.gatype((Scalar, Twist, Twist))


def point(xy: np.ndarray) -> Point:
    """Embed (..., 2) Cartesian positions as unit-weight PGA points."""
    xy = np.asarray(xy, dtype=float)
    return mv.antivector(np.concatenate([xy, np.ones_like(xy[..., :1])], axis=-1))


def coordinates(points: Point) -> np.ndarray:
    """Read x and y coefficients, also for ideal points representing displacements."""
    return points.select_subspace(PGA2D.subspace("yw wx")).kernel


def spring_stiffness(
    lines: Wrench, stiffnesses: Scalar,
) -> tuple[Stiffness, SpringExtension]:
    """Assemble unprestressed axial springs from unit lines of action.

    Orient lines from anchor to attachment so positive extension means
    lengthening. K(q) opposes the restoring wrench, which is -K(q).
    """
    extension: SpringExtension = Twist.regressive(lines)
    stiffness: Stiffness = (lines * extension * stiffnesses).sum(axis=0)
    return stiffness, extension


def body_inertia(points: Point, masses: Scalar) -> Inertia:
    """Sum mass times the open point-velocity-to-momentum construction."""
    return (points.regressive(points.commutator(Twist)) * masses).sum(axis=0)


def normal_modes(stiffness: Stiffness, inertia: Inertia) -> tuple[Twist, np.ndarray]:
    """Return mass-normalized modes and frequencies in Hz, ordered low to high."""
    elastic: EnergyForm = Twist.regressive(stiffness)
    kinetic: EnergyForm = Twist.regressive(inertia)
    return solve_energy_modes(elastic, kinetic)


def solve_energy_modes(elastic: EnergyForm, kinetic: EnergyForm) -> tuple[Twist, np.ndarray]:
    """Numerical boundary: solve the two forms and rewrap eigenvectors as twists."""
    squared, vectors = eigh(elastic.kernel[0], kinetic.kernel[0])
    # A geometric null mode can acquire a tiny eigenvalue through roundoff.
    tolerance = 1e-12 * max(1.0, float(np.max(np.abs(squared))))
    if np.min(squared) < -tolerance:
        raise ValueError("The spring system has negative stiffness.")
    squared = np.where(np.abs(squared) < tolerance, 0.0, squared)
    return mv.bivector(vectors.T), np.sqrt(squared) / (2 * np.pi)


@dataclass(frozen=True)
class Suspension:
    """Geometry and the two mechanical extensors of one planar suspension."""

    body: Point
    attachments: Point
    anchors: Point
    stiffnesses: Scalar
    stiffness: Stiffness
    extension: SpringExtension
    inertia: Inertia


def suspension(angled_spring: bool = False) -> Suspension:
    """A uniform 2-by-1 plate of mass 1, supported by springs of stiffness 6."""
    body = point(np.array([[-1, -.5], [1, -.5], [1, .5], [-1, .5]]))
    attachments = np.array([[-.8, .5], [.8, .5], [1, 0]])
    anchors = np.array([[-.8, 1.55], [.8, 1.55], [1.9, .85]])
    count = 3 if angled_spring else 2
    attachments, anchors = point(attachments[:count]), point(anchors[:count])
    lines: Wrench = anchors.regressive(attachments).normalized()
    constants = mv.scalar(np.full((count, 1), 6.0))
    stiffness, extension = spring_stiffness(lines, constants)

    # Tensor-product two-point Gauss quadrature integrates the plate's mass
    # and quadratic moments exactly, including its polar inertia of 5/12.
    mass_points = point(coordinates(body) / np.sqrt(3))
    inertia = body_inertia(mass_points, mv.scalar(np.full((4, 1), .25)))
    return Suspension(body, attachments, anchors, constants, stiffness, extension, inertia)


def main(
    plot_path: str = str(PLOT_DIR / "stiffness.png"),
    animation_path: str | None = None,
) -> plt.Figure:
    """Plot two spring arrangements and all three modes of each."""
    from examples.mechanics.stiffness_plumbing import PlotCase, draw_modes, save_animation

    cases = []
    for angled in (False, True):
        system = suspension(angled)
        modes, frequencies = normal_modes(system.stiffness, system.inertia)
        # Explicit batch axes: mode x geometric point; no per-point solve.
        body_offsets = coordinates(system.body[None, :].commutator(modes[:, None]))
        scale = .20 / np.linalg.norm(body_offsets, axis=-1).max(axis=-1)
        display_modes = modes * mv.scalar(scale[:, None])
        body_offsets = coordinates(system.body[None, :].commutator(display_modes[:, None]))
        attachment_offsets = coordinates(system.attachments[None, :].commutator(display_modes[:, None]))
        extensions = system.extension(display_modes[:, None]).kernel[..., 0]
        cases.append(PlotCase(
            title="Add an angled spring" if angled else "Two vertical springs",
            description=("All three motions now have a restoring force" if angled else
                         "Sideways motion leaves both springs unchanged to first order"),
            body=coordinates(system.body),
            attachments=coordinates(system.attachments),
            anchors=coordinates(system.anchors),
            frequencies=frequencies,
            body_offsets=body_offsets,
            attachment_offsets=attachment_offsets,
            extensions=extensions,
            labels=("Coupled mode 1", "Coupled mode 2", "Coupled mode 3") if angled else
                   ("Free slide", "Bounce", "Rock"),
        ))
        print(f"{cases[-1].title}: {np.round(frequencies, 4)} Hz")
    figure = draw_modes(cases, plot_path)
    if animation_path:
        save_animation(cases, animation_path)
    return figure


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--animate", action="store_true", help="Also save the normal modes as a GIF.")
    args = parser.parse_args()
    main(animation_path=str(PLOT_DIR / "stiffness.gif") if args.animate else None)
    plt.show()
