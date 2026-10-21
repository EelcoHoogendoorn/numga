"""Nested exceptional root systems and their Coxeter-plane projections."""

from __future__ import annotations

import numpy as np

from examples.math.e_series import core
from examples.math.e_series.core import Scalar, Vector, VectorMap

NAMES = ("E6", "E7", "E8")
RANKS = np.array([6, 7, 8])
PERIODS = np.array([12, 18, 30])


# --- math -----------------------------------------------------------------------------
def family() -> tuple[tuple[VectorMap, Vector, Vector, Vector, Scalar], ...]:
    """Each family's motion, roots, invariant plane frames and turn angles."""
    simple = core.simple_roots()                                              # [mirrors] Vector

    def construct(n_rank: int, n_steps: int) -> tuple[VectorMap, Vector, Vector, Vector, Scalar]:
        action, roots = core.root_system(simple[:n_rank], n_steps)             # Vector <- Vector, [steps, orbits] Vector
        horizontal, vertical, angles = core.rotation_planes(action, n_rank)   # [planes] Vector, Vector, Scalar
        return action, roots, horizontal, vertical, angles

    return tuple(construct(n_rank, n_steps) for n_rank, n_steps in zip(RANKS, PERIODS))


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_figure
    from examples.math.e_series import render

    cases = family()
    root_systems = tuple(roots for _, roots, _, _, _ in cases)
    _, _, horizontal, vertical, _ = cases[-1]

    save_figure(render.draw_diagram(core.simple_roots(), RANKS, NAMES), "e_series_diagram")
    save_figure(render.draw_nested_roots(root_systems, horizontal[0], vertical[0], NAMES),
                "e_series_nested")

    # checks
    # The orbits hold 72, 126 and 240 roots, each of squared length two.
    for roots, n_roots in zip(root_systems, (72, 126, 240)):
        assert np.prod(roots.shape) == n_roots
        np.testing.assert_allclose(roots.scalar_norm_squared().to_array(), 2, atol=1e-10)


if __name__ == "__main__":
    main()
