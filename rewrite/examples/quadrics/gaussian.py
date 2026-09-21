"""Fit a Gaussian and its 1σ quadric directly from homogeneous point moments.

The inverse second moment gives squared Mahalanobis distance plus one on
unit-weight points. Exponentiating gives the Gaussian; subtracting the weight
dyad twice gives its 1σ contour as a homogeneous zero locus. No origin or
principal-axis frame is supplied. This example draws the construction in PGA2D.

Run from rewrite/ with PYTHONPATH=src:. python -m examples.quadrics.gaussian.
"""

from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from numga import NumpyContext
from numga.algebras import PGA2D

from examples import PLOT_DIR


# --- scenario algebra ------------------------------------------------------
ga = PGA2D
ctx = NumpyContext(ga)
mv = ctx.multivector
Point = ga.gatype.antivector()
Plane = ga.gatype.vector()
Scalar = ga.gatype.scalar()


# --- math ------------------------------------------------------------------
def fit_gaussian(points: Point, samples: Point) -> tuple[Scalar, Scalar]:
    """Fit unit-weight points and evaluate the Gaussian and its 1σ level set.

    Samples are unit-weight query points in the same affine chart. The returned
    density has peak one; the level vanishes at Mahalanobis distance one.
    """
    # Leave the plane slot open: each point contributes a rank-one Plane -> Point map.
    # Its mean contains both the cloud's location and its spread, without centering.
    moment = (points * (Plane & points)).mean(axis=0)
    precision = moment.inverse()

    # The inverse moment evaluates to 1 + squared Mahalanobis distance.
    # Removing the constant gives a Gaussian with peak density one.
    squared_distance = (precision(samples) & samples) - 1
    density = (-0.5 * squared_distance).exp()

    # Recover the weight plane from the data: weight & p = 1 for every sample.
    # Its dyad evaluates to that constant squared; subtract twice to make d² = 1
    # the zero locus of a Point -> Plane polarity. 
    weight = precision(points.mean(axis=0))
    quadric = precision - 2 * weight * (weight & Point)
    level = quadric(samples) & samples
    return density, level


# --- plumbing: construction and coordinate readout --------------------------
def point(xy: np.ndarray) -> Point:
    """Embed Cartesian coordinates as unit-weight PGA points."""
    return mv.yw * xy[..., 0] + mv.wx * xy[..., 1] + mv.xy


def euclidean(points: Point) -> np.ndarray:
    coordinates = points.cast(ga.subspace("yw wx xy")).kernel
    return coordinates[..., :2] / coordinates[..., 2:]


# --- plotting --------------------------------------------------------------
def draw(points: Point, pixels: Point, density: Scalar, level: Scalar) -> plt.Figure:
    xy, sample_xy = euclidean(pixels), euclidean(points)
    fig, ax = plt.subplots(figsize=(8, 7), dpi=140, layout="constrained")
    field = ax.pcolormesh(xy[..., 0], xy[..., 1], density.to_array(),
                         shading="auto", cmap="Blues", vmin=0, vmax=1, rasterized=True)
    ax.scatter(sample_xy[:, 0], sample_xy[:, 1], s=10, c="#26364a", alpha=0.4,
               linewidths=0, label="Point cloud")
    ax.contour(xy[..., 0], xy[..., 1], level.to_array(), levels=[0],
               colors=["#e56b24"], linewidths=2.2)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color="#e56b24", lw=2.2))
    ax.legend(handles, labels + ["1σ quadric: Q(p) & p = 0"], loc="upper right")
    ax.set(xlabel="x", ylabel="y", aspect="equal",
           xlim=(xy[0, 0, 0], xy[0, -1, 0]), ylim=(xy[0, 0, 1], xy[-1, 0, 1]),
           title="A Gaussian and its 1σ quadric from point moments")
    fig.colorbar(field, ax=ax, shrink=0.72, label="Density / peak density")
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / f"gaussian_{datetime.now():%Y%m%d_%H%M%S_%f}.png"
    fig.savefig(path)
    print(f"Figure saved to {path}")
    return fig


# --- scenario --------------------------------------------------------------
def main() -> plt.Figure:
    # Inputs: a stretched cloud, moved as a batch, and points covering the plot.
    rng = np.random.default_rng(4)
    points = point(rng.normal(size=(400, 2)) * [1.5, 0.5])
    placement = (mv.xw * 0.6 - mv.yw * 0.3).exp() * (mv.xy * 0.3).exp()
    points = placement >> points
    pixels = point(np.stack(np.meshgrid(np.linspace(-5, 6, 320),
                                       np.linspace(-5, 4, 320)), axis=-1))

    density, level = fit_gaussian(points, pixels)
    return draw(points, pixels, density, level)


if __name__ == "__main__":
    main()
    plt.show()
