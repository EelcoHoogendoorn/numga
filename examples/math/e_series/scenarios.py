"""Nested exceptional root systems and their Coxeter-plane projections."""

from __future__ import annotations

import numpy as np

from examples.math.e_series import core
from examples.math.e_series.core import Plane, PlaneMap, Scalar

names = ("E6", "E7", "E8")
ranks = np.array([6, 7, 8])
periods = np.array([12, 18, 30])


# --- math -----------------------------------------------------------------------------
def family() -> tuple[tuple[PlaneMap, Plane, Plane, Plane, Scalar], ...]:
    """Each system's simple roots generate its Coxeter motion, root orbits, invariant plane frames, and turn angles."""
    simple = core.simple_roots()                                              # [8] Plane+
    systems = tuple(core.root_system(simple[:n_rank], n_steps) for n_rank, n_steps in zip(ranks, periods))
    return tuple((action, roots, *core.rotation_planes(action, n_rank))
                 for (action, roots), n_rank in zip(systems, ranks))


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_figure
    from examples.math.e_series import render

    cases = family()
    root_systems = tuple(roots for _, roots, _, _, _ in cases)
    _, _, plane_x, plane_y, _ = cases[-1]

    save_figure(render.diagram_figure(core.simple_roots(), ranks, names), "e_series_diagram")
    save_figure(render.nested_roots_figure(root_systems, plane_x[0], plane_y[0], names), "e_series_nested")

    # checks
    # The orbits hold 72, 126 and 240 roots, each of squared length two.
    for roots, n_roots in zip(root_systems, (72, 126, 240)):
        assert np.prod(roots.shape) == n_roots
        assert ((roots.scalar_norm_squared() - 2).abs() < 1e-10).all()


if __name__ == "__main__":
    main()
