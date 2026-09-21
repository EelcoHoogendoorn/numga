"""Compare two motor fits on the same corresponding PGA3D points.

The centered sandwich alignment gives a Cartesian least-squares rotation;
matching the centroids supplies translation. The one-sided equation
q M - M p = 0 fits rotation and translation together using a coefficient
least-squares objective, followed by motor normalization. With noisy data
these are different objectives.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from numga import NumpyContext
from numga.algebras import PGA3D

from examples import PLOT_DIR


# --- scenario algebra ------------------------------------------------------
ga = PGA3D
ctx = NumpyContext(ga)
mv = ctx.multivector
Point = ga.gatype.antivector()
Motor = ga.gatype.rotor()
# The Euclidean rotation subgroup, fixing the chosen PGA origin.
Rotor = ga.gatype.from_blades("1 yz zx xy")
Vector = ga.gatype.from_blades("x y z")


# --- math ------------------------------------------------------------------
def fit_motor(source: Point, target: Point) -> Motor:
    """Fit a motor by the coefficient residual of the one-sided equations."""
    source, target = source.normalized(), target.normalized()
    # q M = M p leaves the unknown motor in a single linear slot.
    residual = target * Motor - Motor * source

    # Transpose pairs coefficients, retaining errors in ideal components too.
    # The degenerate PGA scalar product would discard translation information.
    misfit = residual.transpose()(residual).sum(axis=0)
    values, motors = misfit.eigh()
    return motors[values.argmin()].normalized()


def fit_rotor(source: Vector, target: Vector) -> Rotor:
    """Maximize sandwich alignment of corresponding Euclidean vectors."""
    alignment = target.scalar_product(Rotor >> source).sum(axis=0)
    values, rotors = ((alignment + alignment.transpose()) * 0.5).eigh()
    return rotors[values.argmax()].normalized()


def fit_motor_alignment(source: Point, target: Point) -> Motor:
    """Fit a rigid pose by centered alignment and centroid matching."""
    source, target = source.normalized(), target.normalized()
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)

    # Point differences are ideal points. Their duals carry the Euclidean
    # displacements; the spatial slot discards the homogeneous weight component.
    source_vectors = (source - source_mean).dual().select_subspace(Vector.output_subspace)
    target_vectors = (target - target_mean).dual().select_subspace(Vector.output_subspace)
    rotation = fit_rotor(source_vectors, target_vectors)

    # A product of point reflections translates by twice their separation.
    # Its square root carries the rotated source centroid onto the target's.
    translation = (target_mean * (rotation >> source_mean).inverse()).square_root()
    return translation * rotation


# --- plumbing: sampling and coordinate readout ------------------------------
def point(xyz: np.ndarray) -> Point:
    return mv.yzw * xyz[..., 0] + mv.zxw * xyz[..., 1] + mv.xyw * xyz[..., 2] + mv.zyx


def cloud(n: int, rng: np.random.Generator) -> Point:
    return point(rng.normal(size=(n, 3)) * [2.0, 1.0, 0.5])


def jitter(points: Point, sigma: float, rng: np.random.Generator) -> Point:
    """Move each point by an independent Gaussian translation."""
    noise = rng.normal(scale=sigma, size=(*points.shape, 3))
    translation = (mv.xw * noise[..., 0] + mv.yw * noise[..., 1] + mv.zw * noise[..., 2]) * 0.5
    return translation.exp() >> points


def coordinates(points: Point) -> np.ndarray:
    values = points.cast(ga.subspace("yzw zxw xyw zyx")).kernel
    return values[..., :3] / values[..., 3:]


# --- plotting --------------------------------------------------------------
def draw_registration(source: Point, target: Point, aligned: Point,
                      title: str, plot_path: Path) -> plt.Figure:
    fig = plt.figure(figsize=(8, 7), dpi=120, layout="constrained")
    ax = fig.add_subplot(projection="3d")
    source_xyz, target_xyz, aligned_xyz = map(coordinates, (source, target, aligned))
    ax.scatter(*source_xyz.T, color="#94a3b8", s=10, label="source")
    ax.scatter(*target_xyz.T, color="#0284c7", s=18, marker="x", label="target")
    ax.scatter(*aligned_xyz.T, color="#f43f5e", s=10, label="aligned source")
    for p, q in zip(aligned_xyz, target_xyz):
        ax.plot(*np.stack([p, q]).T, color="#f43f5e", linewidth=0.5, alpha=0.6)
    ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])
    ax.legend(loc="upper left", fontsize=8)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, bbox_inches="tight")
    print(f"Figure saved to {plot_path}")
    return fig


# --- scenarios -------------------------------------------------------------
def correspondences() -> tuple[Point, Point]:
    """The same seeded, noisy rigid correspondences for both fitting methods."""
    rng = np.random.default_rng(0)
    source = cloud(60, rng)
    truth = (mv.xw * 0.75 - mv.yw * 0.25 + mv.zw).exp() * (mv.xy * 0.4 - mv.yz * 0.3 + mv.zx * 0.7).exp()
    target = jitter(truth >> source, 0.02, rng)

    return source, target


def sandwich_alignment() -> plt.Figure:
    """Cartesian least squares: center, fit rotation, then match centroids."""
    source, target = correspondences()
    estimate = fit_motor_alignment(source, target)
    return draw_registration(source, target, estimate >> source, "Centered sandwich alignment",
                             PLOT_DIR / "registration_sandwich_alignment.png")


def one_sided_residual() -> plt.Figure:
    """Coefficient least squares: fit and normalize a motor in one eigenproblem."""
    source, target = correspondences()
    estimate = fit_motor(source, target)
    return draw_registration(source, target, estimate >> source, "One-sided motor residual",
                             PLOT_DIR / "registration_one_sided_residual.png")


if __name__ == "__main__":
    sandwich_alignment()
    one_sided_residual()
    plt.show()
