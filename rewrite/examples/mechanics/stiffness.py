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

Run from rewrite/ with PYTHONPATH=src:. python -m examples.mechanics.stiffness.
Add --animate to also save plots/stiffness.gif.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

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


def point(xy: np.ndarray) -> Point:
    """Embed (..., 2) Cartesian positions as PGA points (dual of homogeneous vector)."""
    return (mv.x * xy[..., 0] + mv.y * xy[..., 1] + mv.w).dual()


def coordinates(points: Point) -> np.ndarray:
    """Read x and y coefficients, also for ideal points representing displacements."""
    return points.select_subspace(PGA2D.subspace("yw wx")).kernel


@dataclass(frozen=True)
class Suspension:
    """Geometry and masses prepared for one planar suspension."""

    body: Point
    attachments: Point
    anchors: Point
    stiffnesses: Scalar
    mass_points: Point
    masses: Scalar


def suspension(angled_spring: bool = False) -> Suspension:
    """A uniform 2-by-1 plate of mass 1, supported by springs of stiffness 6."""
    body = point(np.array([[-1, -.5], [1, -.5], [1, .5], [-1, .5]]))
    attachments = np.array([[-.8, .5], [.8, .5], [1, 0]])
    anchors = np.array([[-.8, 1.55], [.8, 1.55], [1.9, .85]])
    count = 3 if angled_spring else 2
    attachments, anchors = point(attachments[:count]), point(anchors[:count])
    constants = mv.scalar(np.full((count, 1), 6.0))

    # Tensor-product two-point Gauss quadrature integrates the plate's mass
    # and quadratic moments exactly, including its polar inertia of 5/12.
    mass_points = point(coordinates(body) / np.sqrt(3))
    masses = mv.scalar(np.full((4, 1), .25))
    return Suspension(body, attachments, anchors, constants, mass_points, masses)


def mode_case(system: Suspension, values: Scalar, body_offsets: Point,
              attachment_offsets: Point, extensions: Scalar, angled: bool):
    """Read mode geometry into plotting arrays and choose a visible amplitude."""
    from examples.mechanics.stiffness_plumbing import PlotCase

    frequencies = np.sqrt(np.maximum(values.kernel[..., 0], 0.0)) / (2 * np.pi)
    offsets = coordinates(body_offsets)
    scale = .20 / np.linalg.norm(offsets, axis=-1).max(axis=-1)
    return PlotCase(
        title="Add an angled spring" if angled else "Two vertical springs",
        description=("All three motions now have a restoring force" if angled else
                     "Sideways motion leaves both springs unchanged to first order"),
        body=coordinates(system.body), attachments=coordinates(system.attachments),
        anchors=coordinates(system.anchors), frequencies=frequencies,
        body_offsets=offsets * scale[:, None, None],
        attachment_offsets=coordinates(attachment_offsets) * scale[:, None, None],
        extensions=extensions.kernel[..., 0] * scale[:, None],
        labels=("Coupled mode 1", "Coupled mode 2", "Coupled mode 3") if angled else
               ("Free slide", "Bounce", "Rock"),
    )


def main(
    plot_path: str = str(PLOT_DIR / "stiffness.png"),
    animation_path: str = "",
) -> plt.Figure:
    """Plot two spring arrangements and all three modes of each."""
    from examples.mechanics.stiffness_plumbing import draw_modes, save_animation

    systems = (suspension(False), suspension(True))
    cases = []
    for angled, system in zip((False, True), systems):
        # A spring measures extension by pairing its line with an open twist.
        # Sum its force response to get stiffness; sum point momenta to get inertia.
        lines: Wrench = (system.anchors & system.attachments).normalized()
        extension: SpringExtension = Twist & lines
        stiffness: Stiffness = (lines * extension * system.stiffnesses).sum(axis=0)
        inertia: Inertia = (system.mass_points & system.mass_points.commutator(Twist) * system.masses).sum(axis=0)

        # Pair the response maps with an open twist to obtain the two energy forms.
        # The generalized eigenvectors are mass-normalized vibration modes.
        values, modes = (Twist & stiffness).eigh(Twist & inertia)
        body_offsets = system.body[None, :].commutator(modes[:, None])
        attachment_offsets = system.attachments[None, :].commutator(modes[:, None])
        extensions = extension(modes[:, None])
        cases.append(mode_case(system, values, body_offsets, attachment_offsets, extensions, angled))

    figure = draw_modes(cases, plot_path)
    if animation_path:
        save_animation(cases, animation_path)
    return figure


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--animate", action="store_true", help="Also save the normal modes as a GIF.")
    args = parser.parse_args()
    main(animation_path=str(PLOT_DIR / "stiffness.gif") if args.animate else "")
    plt.show()
