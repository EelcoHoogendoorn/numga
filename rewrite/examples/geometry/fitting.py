"""Least-squares fitting of PGA3D primitives to points, in one pattern.

A point, a line and a plane are fitted to noisy samples with the same three lines: the
join of the samples with the unknown left open, that residual squared and summed into a
quadratic form, and its smallest unit eigenvector. Unit is the unknown's own reverse
product, which is degenerate exactly on the coefficients least squares leaves free. The
roles swap freely: a point fitted to a bundle of lines is their point of closest approach.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples import PLOT_DIR
from examples.geometry.fitting_plumbing import (
    Line,
    Plane,
    Point,
    bundle,
    cloud,
    jitter,
    line_caps,
    mv,
    new_figure,
    patch,
    patch_edges,
    render_line_fit,
    render_bundle_fit,
    render_plane_fit,
    render_point_fit,
    same_element,
    segment,
    smallest_eigenvector,
)


def main(plot_path: str = str(PLOT_DIR / "fitting.png")) -> plt.Figure:
    """Fit a point, a line and a plane to noisy samples and draw them."""
    rng = np.random.default_rng(0)

    # -----------------------------------------------------------------------
    # 1. Ground truth: one pose applied to the canonical primitives
    # -----------------------------------------------------------------------
    # The point zyx, the line xz and the plane z are basis blades. A motor moves all three
    # and their samples at once, so the fits have a known answer to be checked against.
    pose = (mv.xw * 0.4 + mv.yw * -0.3 + mv.zw * 0.6).exp() * (mv.yz * 0.3).exp() * (mv.xy * 0.5).exp()

    # -----------------------------------------------------------------------
    # 2. Plane
    # -----------------------------------------------------------------------
    # The join of a sample with a plane is a scalar, the signed distance, and it vanishes
    # iff the sample lies on the plane. With the plane left open that is one linear map per
    # sample, from candidate planes to residuals. Squared and summed, the residuals are a
    # quadratic form in the plane: the normal matrix, assembled without writing an entry.
    # Its smallest unit eigenvector is the fit. The unit condition is the plane's own
    # reverse product, which only sees the normal: the offset is free, as it should be.
    plane_points = jitter(pose >> patch(200, 2.0, rng), 0.05, rng)
    residual = plane_points.regressive(Plane)
    misfit = (residual.reverse() | residual).sum()
    plane = smallest_eigenvector(misfit)
    assert same_element(plane, pose >> mv.z, atol=0.02)

    # -----------------------------------------------------------------------
    # 3. Line: the same lines with a line in the slot
    # -----------------------------------------------------------------------
    # The join of a sample with a line is a plane, and the line's reverse product only
    # sees its direction: the moment is free. Nothing else changes.
    line_points = jitter(pose >> segment(120, 2.0), 0.05, rng)
    residual = line_points.regressive(Line)
    misfit = (residual.reverse() | residual).sum()
    line = smallest_eigenvector(misfit)
    assert same_element(line, pose >> mv.xz, atol=0.02)

    # The moment was never constrained to be a moment, yet L ∧ L = 0 holds: the free
    # coefficients settle where the line passes through the centroid, which forces it.
    np.testing.assert_allclose(line.wedge(line).kernel / float(np.abs(line.kernel).max()) ** 2, 0.0, atol=1e-14)

    # -----------------------------------------------------------------------
    # 4. Point: the same lines with a point in the slot
    # -----------------------------------------------------------------------
    # The join of a sample with a point is a line, and a point's reverse product only sees
    # its weight: the whole position is free. The fit is the centroid, which for
    # unit-weight points is simply their sum.
    points = jitter(pose >> cloud(200, 0.5, rng), 0.05, rng)
    residual = points.regressive(Point)
    misfit = (residual.reverse() | residual).sum()
    centroid = smallest_eigenvector(misfit)
    assert same_element(centroid, points.sum(), atol=1e-10)

    # -----------------------------------------------------------------------
    # 5. Point to lines: the roles swapped
    # -----------------------------------------------------------------------
    # Nothing said the samples had to be points. A bundle of lines in the data and a point
    # in the slot is the same three lines, and the fit is the point of closest approach:
    # triangulation, when the lines are rays from cameras.
    rays = pose >> bundle(30, 0.05, rng)
    residual = rays.regressive(Point)
    misfit = (residual.reverse() | residual).sum()
    meet = smallest_eigenvector(misfit)
    assert same_element(meet, pose >> mv.zyx, atol=0.05)

    # -----------------------------------------------------------------------
    # 6. Draw: read every fit out by meeting it with lines and planes
    # -----------------------------------------------------------------------
    fig, axes = new_figure()
    render_point_fit(axes[0], points, pose >> mv.zyx, centroid)
    render_line_fit(axes[1], line_points, pose >> mv.xz, line, caps=pose >> line_caps(2.5))
    render_plane_fit(axes[2], plane_points, pose >> mv.z, plane, edges=pose >> patch_edges(2.0))
    render_bundle_fit(axes[3], rays, pose >> mv.zyx, meet, half_length=2.0)
    for ax in axes:
        ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    if plot_path:
        plt.savefig(plot_path, bbox_inches="tight")
        print(f"Figure saved to {plot_path}")
    return fig


if __name__ == "__main__":
    main()
    plt.show()
