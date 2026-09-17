"""2D Projective Geometric Algebra (PGA2D) in R_{2,0,1}.

Demonstrates idiomatic 2D Euclidean geometry in PGA2D:
- Points are antivectors (homogeneous coordinates (x, y, 1))
- Lines are vectors (nx*x + ny*y - d*w = 0)
- Join of points is the regressive product: `p1.regressive(p2)`
- Meet of lines is the exterior wedge product: `l1.wedge(l2)`
- Rotations are bivector point exponentials: `(-0.5 * angle * center).exp()`
- Translations are ideal line wedges: `(-0.5 * (tx*wx + ty*wy)).exp()`
- Rigid motion is the sandwich product: `motor.sandwich(geometry)`
- Multi-object transformation compiles to an arity-1 Extensor operator
  by sandwiching the antivector subspace hole:
      `M = motor.sandwich(spaces.antivector())`
      `transformed = M(points)`
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext
from numga.algebras import PGA2D
from examples import PLOT_DIR

context = NumpyContext(PGA2D)
spaces = PGA2D.subspace
mv = context.multivector

Point = PGA2D.gatype.antivector()
Line = PGA2D.gatype.vector()
Motor = PGA2D.gatype.rotor()

# Coordinate axes: lines through origin with normals along x and y
y_axis = mv.vector([1.0, 0.0, 0.0])  # line x = 0
x_axis = mv.vector([0.0, 1.0, 0.0])  # line y = 0
line_at_infinity = mv.vector([0.0, 0.0, 1.0])  # ideal line w = 0

# Canonical origin: meet of coordinate axes (x ^ y = xy bivector)
origin = y_axis.wedge(x_axis)


def point(px: float | np.ndarray, py: float | np.ndarray) -> Point:
    """Construct an affine point (px, py, 1) as a PGA antivector."""
    px, py = np.asarray(px), np.asarray(py)
    ones = np.ones_like(px)
    coords = np.stack([px, py, ones], axis=-1)
    return mv.antivector(coords)


def ideal_point(dx: float | np.ndarray, dy: float | np.ndarray) -> Point:
    """Construct an ideal point at infinity (dx, dy, 0) as a PGA antivector."""
    dx, dy = np.asarray(dx), np.asarray(dy)
    zeros = np.zeros_like(dx)
    coords = np.stack([dx, dy, zeros], axis=-1)
    return mv.antivector(coords)


def line(nx: float, ny: float, d: float) -> Line:
    """Construct an oriented line nx*x + ny*y - d*w = 0 as a PGA vector."""
    return mv.vector([nx, ny, -d])


def translator(tx: float | np.ndarray, ty: float | np.ndarray) -> Motor:
    """Construct a translation motor by displacement (tx, ty)."""
    tx, ty = np.asarray(tx), np.asarray(ty)
    zeros = np.zeros_like(tx)
    displacement = mv.vector(np.stack([tx, ty, zeros], axis=-1))
    generator = line_at_infinity.wedge(displacement) * -0.5
    return generator.exp()


def rotor(angle: float, center: Point = origin) -> Motor:
    """Construct a rotation motor by angle around a center point (default: origin)."""
    return (center * (-angle / 2.0)).exp()


def draw_polygons(polygons: Point, centre: Point, save_path: str) -> None:
    """Read polygon coordinates and draw the family of rotated outlines."""
    import matplotlib.pyplot as plt
    xy_subspace = spaces("yw wx")
    coords = polygons.select_subspace(xy_subspace).kernel
    centre_xy = centre.select_subspace(xy_subspace).kernel
    fig, ax = plt.subplots(figsize=(6, 6))
    for polygon in coords:
        ax.plot(polygon[:, 0], polygon[:, 1])
    ax.scatter(*centre_xy, color="red", label="Center of rotation")
    ax.axis("equal"); ax.grid(True, alpha=0.3); ax.legend()
    ax.set_title("PGA2D: Affine Operator Transformation")
    if save_path:
        fig.savefig(save_path, dpi=150)


def run_pga_plot(save_path: str = str(PLOT_DIR / "pga2d_rotation.png")) -> None:
    """Rotate a polygon around an off-center point using a compiled affine operator."""
    a = np.linspace(0, np.pi * 2, 7, endpoint=True)
    poly = point(np.cos(a), np.sin(a))
    p = point(-10.0, 1.0)

    angles = mv.scalar(np.linspace(0, 1, 10)[:, None])
    # The point is the rotation generator; leave the passenger open to compile its map.
    motors = (p * (-angles / 2)).exp()
    maps = motors >> Point
    transformed = maps[:, None](poly)
    draw_polygons(transformed, p, save_path)


if __name__ == "__main__":
    run_pga_plot()
