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

Run from rewrite/ with PYTHONPATH=src:. python -m examples.mechanics.modes.
Add --animate to also save plots/modes.gif.
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
Wrench = PGA2D.gatype.antibivector()
SpringExtension = PGA2D.gatype((Scalar, Twist))
Stiffness = PGA2D.gatype((Wrench, Twist))
Inertia = PGA2D.gatype((Wrench, Twist))


def point(xy: np.ndarray) -> Point:
    """Embed (..., 2) Cartesian positions as PGA points (dual of homogeneous vector)."""
    return (mv.x * xy[..., 0] + mv.y * xy[..., 1] + mv.w).dual()


@dataclass(frozen=True)
class Suspension:
    """Geometry and masses prepared for one planar suspension."""

    body: Point                     # [4] Point
    attachments: Point              # [n_springs] Point
    anchors: Point                  # [n_springs] Point
    spring_constants: np.ndarray    # [n_springs] float
    mass_points: Point              # [4] Point
    masses: np.ndarray              # [4] float

    @property
    def stiffnesses(self) -> np.ndarray:
        """Alias for spring_constants for backwards compatibility."""
        return self.spring_constants


def suspension(angled_spring: bool = False) -> Suspension:
    """A uniform 2-by-1 plate of mass 1, supported by springs of stiffness 6."""
    body_xy = np.array([[0.0, 0.5], [2.0, 0.5], [2.0, 1.5], [0.0, 1.5]])          # [4, 2] float (center at (1, 1))
    attachments_xy = np.array([[0.2, 1.5], [1.8, 1.5], [2.0, 1.0]])              # [3, 2] float
    anchors_xy = np.array([[0.2, 2.55], [1.8, 2.55], [2.9, 1.85]])               # [3, 2] float
    count = 3 if angled_spring else 2
    body = point(body_xy)                                                         # [4] Point
    attachments = point(attachments_xy[:count])                                   # [n_springs] Point
    anchors = point(anchors_xy[:count])                                           # [n_springs] Point
    spring_constants = np.full(count, 6.0)                                        # [n_springs] float

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
) -> PlotCase:
    """Format mode geometry into a PlotCase."""
    from examples.mechanics.modes.render import PlotCase

    frequencies = np.sqrt(np.maximum(values.kernel[..., 0], 0.0)) / (2 * np.pi)  # [3] float
    return PlotCase(
        body=body,
        attachments=attachments,
        anchors=anchors,
        frequencies=frequencies,
        body_offsets=body_offsets,
        attachment_offsets=attachment_offsets,
        extensions=extensions,
    )


def main(
    plot_path: str = str(PLOT_DIR / "modes.png"),
    animation_path: str = "",
) -> plt.Figure:
    """Plot two spring arrangements and all three modes of each."""
    from examples.mechanics.modes.render import draw_modes, save_animation

    systems = (suspension(False), suspension(True))
    cases = []
    for angled, system in zip((False, True), systems):
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

        cases.append(mode_case(
            body=system.body,
            attachments=system.attachments,
            anchors=system.anchors,
            values=values,
            body_offsets=body_offsets,
            attachment_offsets=attachment_offsets,
            extensions=extensions,
        ))

    figure = draw_modes(cases, title="Normal Modes: Baseline (top) vs Coupled (bottom)", plot_path=plot_path)
    if animation_path:
        save_animation(cases, animation_path)
    return figure


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--animate", action="store_true", help="Also save the normal modes as a GIF.")
    args = parser.parse_args()
    main(animation_path=str(PLOT_DIR / "modes.gif") if args.animate else "")
