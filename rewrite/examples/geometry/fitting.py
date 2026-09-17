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
    draw_fits,
    Line,
    Plane,
    Point,
    bundle,
    cloud,
    jitter,
    mv,
    patch,
    same_element,
    segment,
    smallest_finite,
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

    plane_points = jitter(pose >> patch(200, 2.0, rng), 0.05, rng)
    line_points = jitter(pose >> segment(120, 2.0), 0.05, rng)
    points = jitter(pose >> cloud(200, 0.5, rng), 0.05, rng)
    rays = pose >> bundle(30, 0.05, rng)

    # `mv.rotor() >> Space` evaluates the identity operator / basis on that subspace.
    # Pairing it with `Space` via reverse and inner product forms the subspace's metric /
    # Gram matrix (B(x, x) = ~x | x) for the generalized eigenvalue problem (A v = λ B v).
    plane_norm = (mv.rotor() >> Plane).reverse() | Plane
    line_norm = (mv.rotor() >> Line).reverse() | Line
    point_norm = (mv.rotor() >> Point).reverse() | Point

    # -----------------------------------------------------------------------
    # 2. Plane
    # -----------------------------------------------------------------------
    # The join of a sample with a plane is a scalar, the signed distance, and it vanishes
    # iff the sample lies on the plane. With the plane left open that is one linear map per
    # sample, from candidate planes to residuals. Squared and summed, the residuals are a
    # quadratic form in the plane: the normal matrix, assembled without writing an entry.
    # Its smallest unit eigenvector is the fit. The unit condition is the plane's own
    # reverse product, which only sees the normal: the offset is free, as it should be.
    residual = plane_points.regressive(Plane)
    misfit = (residual.reverse() | residual).sum()
    values, vectors = ((misfit + misfit.transpose()) * 0.5).eig(plane_norm)
    plane = smallest_finite(values, vectors)

    # -----------------------------------------------------------------------
    # 3. Line: the same lines with a line in the slot
    # -----------------------------------------------------------------------
    # The join of a sample with a line is a plane, and the line's reverse product only
    # sees its direction: the moment is free. Nothing else changes.
    residual = line_points.regressive(Line)
    misfit = (residual.reverse() | residual).sum()
    values, vectors = ((misfit + misfit.transpose()) * 0.5).eig(line_norm)
    line = smallest_finite(values, vectors)

    # The moment was never constrained to be a moment, yet L ∧ L = 0 holds: the free
    # coefficients settle where the line passes through the centroid, which forces it.

    # -----------------------------------------------------------------------
    # 4. Point: the same lines with a point in the slot
    # -----------------------------------------------------------------------
    # The join of a sample with a point is a line, and a point's reverse product only sees
    # its weight: the whole position is free. The fit is the centroid, which for
    # unit-weight points is simply their sum.
    residual = points.regressive(Point)
    misfit = (residual.reverse() | residual).sum()
    values, vectors = ((misfit + misfit.transpose()) * 0.5).eig(point_norm)
    centroid = smallest_finite(values, vectors)

    # -----------------------------------------------------------------------
    # 5. Point to lines: the roles swapped
    # -----------------------------------------------------------------------
    # Nothing said the samples had to be points. A bundle of lines in the data and a point
    # in the slot is the same three lines, and the fit is the point of closest approach:
    # triangulation, when the lines are rays from cameras.
    residual = rays.regressive(Point)
    misfit = (residual.reverse() | residual).sum()
    values, vectors = ((misfit + misfit.transpose()) * 0.5).eig(point_norm)
    meet = smallest_finite(values, vectors)

    # -----------------------------------------------------------------------
    # 6. Draw: read every fit out by meeting it with lines and planes
    # -----------------------------------------------------------------------
    fig = draw_fits(points, line_points, plane_points, rays, pose, centroid, line, plane, meet, plot_path)


    # --- checks -------------------------------------------------------------
    assert same_element(plane, pose >> mv.z, atol=0.02)
    assert same_element(line, pose >> mv.xz, atol=0.02)
    np.testing.assert_allclose(line.wedge(line).kernel / float(np.abs(line.kernel).max()) ** 2, 0.0, atol=1e-14)
    assert same_element(centroid, points.sum(), atol=1e-10)
    assert same_element(meet, pose >> mv.zyx, atol=0.05)

    return fig


if __name__ == "__main__":
    main()
    plt.show()
