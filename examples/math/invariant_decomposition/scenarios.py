"""The scenes split a random rotation of six-dimensional space into its three planes and trace a point it turns over one
turn of its slowest plane, and split a random rigid motion of six-dimensional space into its three placed planes."""

from __future__ import annotations

import numpy as np

from examples.math.invariant_decomposition import core

# The seed of the random rotation, starting point and rigid motion; the times the point is traced at, the moments the
# animation shows, and the duration of each in milliseconds.
SEED = 0
N_TIMES = 600
N_FRAMES = 120
DURATION_MS = 50


# --- plumbing -------------------------------------------------------------------------
def samples(seed: int) -> tuple[core.Bivector, core.Antivector, core.ProjectiveBivector]:
    """A random rotation's bivector, a random point at unit distance from the origin, and a random rigid motion's
    bivector."""
    rng = np.random.default_rng(seed)
    bivector = core.mv(core.Bivector, rng.normal(size=15))                     # [] Bivector+
    start = core.mv(core.Antivector, rng.normal(size=6)).normalized()          # [] Antivector-
    motion = core.pmv(core.ProjectiveBivector, rng.normal(size=21))            # [] Bivector+
    return bivector, start, motion


# --- math -----------------------------------------------------------------------------
def tangle(bivector: core.Bivector, start: core.Antivector, n_times: int) -> tuple[core.Bivector, core.Antivector, core.Antivector]:
    """The rotation's planes, the start turned by it over one turn of its slowest plane, and the turned point's part
    in each plane, which turns there on a circle."""
    squares, parts = core.from_spectrum(bivector, core.Antivector)             # [3] Scalar+, [3] Bivector+
    period = 2 * np.pi / (-squares[squares.argmax()]).square_root()            # [] Scalar+
    times = period * np.linspace(0, 1, n_times)                                # [times] Scalar+
    rotors = core.rotor(squares, parts, times)                                 # [times] Rotor+
    points = rotors >> start                                                   # [times] Antivector-
    # A point's acceleration in one plane alone is that plane's square times the point's part in it.
    circles = points.commutator(parts[:, None]).commutator(parts[:, None]) / squares[:, None]   # [3, times] Antivector-
    return parts, points, circles


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.math.invariant_decomposition import render

    bivector, start, motion = samples(SEED)
    parts, points, circles = tangle(bivector, start, N_TIMES)
    save_figure(render.draw(points, circles, parts, N_TIMES - 1), "invariant_decomposition")
    save_animation(render.animate(points, circles, parts, N_FRAMES), "invariant_decomposition", DURATION_MS)
    planes = core.placed(motion)                                               # [3] Bivector+

    # checks
    # The planes sum to the bivector, commute, and are simple; the two ways give the same squares and the same
    # planes. The product of the planes' rotors is the exponential of half the bivector, the point keeps its length,
    # and in each plane it stays on a circle.
    squares, _ = core.from_spectrum(bivector, core.Antivector)
    wedge_squares, wedge_parts = core.from_wedge_powers(bivector)
    np.testing.assert_allclose((parts.sum(axis=0) - bivector).kernel, 0.0, atol=1e-11)
    np.testing.assert_allclose(parts[:, None].commutator(parts[None]).kernel, 0.0, atol=1e-11)
    np.testing.assert_allclose((parts ^ parts).kernel, 0.0, atol=1e-10)
    np.testing.assert_allclose((wedge_squares - squares).kernel, 0.0, atol=1e-11)
    np.testing.assert_allclose((wedge_parts - parts).kernel, 0.0, atol=1e-11)
    np.testing.assert_allclose((core.rotor(squares, parts, core.mv.scalar([1])) - (bivector / 2).exp()).kernel, 0.0, atol=1e-6)
    np.testing.assert_allclose(points.scalar_norm_squared().kernel, 1.0, atol=1e-6)
    radii = circles.scalar_norm_squared()                                      # [3, times] Scalar+
    np.testing.assert_allclose((radii - radii[:, :1]).kernel, 0.0, atol=1e-6)
    # The placed planes of the motion are simple and commute with one another, and in six dimensions they add up to
    # the whole motion.
    np.testing.assert_allclose((planes ^ planes).kernel, 0.0, atol=1e-11)
    np.testing.assert_allclose(planes[:, None].commutator(planes[None]).kernel, 0.0, atol=1e-11)
    np.testing.assert_allclose((motion - planes.sum(axis=-1)).kernel, 0.0, atol=1e-10)


if __name__ == "__main__":
    main()
