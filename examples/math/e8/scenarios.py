"""E8's 240 roots and the four perpendicular planes of its Coxeter rotation."""

from __future__ import annotations

from functools import reduce
from operator import mul

import numpy as np

from numga.extensor import stack

from examples.math.e8 import core
from examples.math.e8.core import Bivector, Rotor, Scalar, Vector


# --- math -----------------------------------------------------------------------------
def root_system() -> tuple[Vector, Vector, Vector, Bivector, Scalar, Rotor, Vector]:
    """The eight root orbits, four oriented plane frames, their winding rates, and their rotor."""
    n_steps = core.coxeter_order
    simple_roots = core.simple_roots()                                         # [mirrors] Vector+
    mirrors = simple_roots.normalized()                                        # [mirrors] Vector+
    coxeter = reduce(mul, mirrors)                                             # [] Rotor+
    action = coxeter >> Vector                                                 # [] Vector+ <- Vector+

    # Prefix reflections pick one seed per orbit; their thirty-step paths exhaust the roots.
    seeds = stack(core.reflection_seeds(simple_roots))                         # [orbits] Vector+
    roots = stack(core.orbit(action, seeds, n_steps))                          # [steps, orbits] Vector+

    # Forward and backward motion average to the cosine of the turn in each invariant plane.
    symmetric = (action + action.adjoint()) / 2                                # [] Vector+ <- Vector+
    cosines, eigenvectors = symmetric.eigh()                                   # [eigenvectors] Scalar+, Vector+

    # Equal eigenvalues occur in pairs. Descending cosines put the slowest plane first.
    plane_x = eigenvectors[::-2]                                               # [planes] Vector+
    plane_y = (action(plane_x) - cosines[::-2] * plane_x).normalized()         # [planes] Vector+
    planes = plane_x ^ plane_y                                                 # [planes] Bivector+
    rates = cosines[::-2].arccos() * n_steps / (2 * np.pi)                     # [planes] Scalar+
    return roots, plane_x, plane_y, planes, rates, coxeter, seeds


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.math.e8 import render

    n_substeps = 6
    n_frames = core.coxeter_order * n_substeps
    duration_ms = 60
    roots, plane_x, plane_y, planes, rates, coxeter, seeds = root_system()
    save_figure(render.draw_roots(roots, plane_x[0], plane_y[0]), "e8_roots")

    # Small steps follow the four commuting turns continuously between the root symmetries.
    generator = -(planes * rates).sum(axis=0) * np.pi / core.coxeter_order     # [] Bivector+
    advance = (generator / n_substeps).exp() >> Vector                         # [] Vector+ <- Vector+
    turning_roots = core.orbit(advance, seeds, n_frames)
    save_animation(render.animate_planes(turning_roots, roots, plane_x, plane_y), "e8_planes", duration_ms)

    # checks
    # The orbits hold all 240 roots, each of squared length two, and the four planes turn 1, 7, 11 and 13
    # times in the thirty steps.
    np.testing.assert_allclose(roots.scalar_norm_squared().to_array(), 2, atol=1e-10)
    assert len(np.unique(np.round(roots.kernel.reshape(-1, 8), decimals=9), axis=0)) == 240
    np.testing.assert_allclose(rates.to_array(), [1, 7, 11, 13], atol=1e-10)


if __name__ == "__main__":
    main()
