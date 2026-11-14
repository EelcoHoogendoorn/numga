"""The scenes split a random rotation of six-dimensional space into its three planes, trace a direction it turns over one
period of its slowest plane, and split a random rigid motion of six-dimensional space into its three placed planes."""

from __future__ import annotations

import numpy as np

from examples.math.invariant_decomposition import core

# The seed of the random rotation, starting direction and rigid motion; the times the orbit is traced at, the frames
# the animation shows, and the duration of each frame in milliseconds.
seed = 0
n_times = 600
n_frames = 150
frame_duration_ms = 40


# --- plumbing -------------------------------------------------------------------------
def samples(seed_value: int) -> tuple[core.Bivector, core.Direction, core.ProjectiveBivector]:
    """A seeded normal draw provides a random 6D rotation bivector (`[] Bivector+`), unit starting direction (`[] Antivector-`), and 6D projective motion (`[] Bivector+`)."""
    rng = np.random.default_rng(seed_value)
    bivector = core.mv(core.Bivector, rng.normal(size=15))                     # [] Bivector+
    start = core.mv.vector(rng.normal(size=6)).dual().normalized()             # [] Antivector-
    motion = core.pmv(core.ProjectiveBivector, rng.normal(size=21))            # [] Bivector+
    return bivector, start, motion


# --- math -----------------------------------------------------------------------------
def orbit_tangle(bivector: core.Bivector, start: core.Direction, n_times: int) -> tuple[core.Scalar, core.Bivector, core.Direction, core.Direction]:
    """Factored plane rotors turn `start` over `n_times` samples of one slowest-plane period, giving the plane squares, commuting planes, orbit (`[n_times] Antivector-`), and plane circles (`[3, n_times] Antivector-`)."""
    squares, parts = core.spectral_decomposition(bivector, core.Direction)     # [3] Scalar+, [3] Bivector+
    period = 2 * np.pi / (-squares[squares.argmax()]).square_root()            # [] Scalar+
    times = period * np.linspace(0, 1, n_times)                                # [n_times] Scalar+
    rotors = core.factored_rotor(squares, parts, times)                        # [n_times] Rotor+
    points = rotors >> start                                                   # [n_times] Antivector-
    # Accelerating a direction in one plane scales its component in that plane by the plane's square.
    circles = points.commutator(parts[:, None]).commutator(parts[:, None]) / squares[:, None]   # [3, n_times] Antivector-
    return squares, parts, points, circles


def main() -> None:
    import matplotlib.pyplot as plt
    from examples.math.invariant_decomposition import render
    from examples.util.animation import save_animation, save_figure

    bivector, start, motion = samples(seed)
    squares, parts, points, circles = orbit_tangle(bivector, start, n_times)
    figure = render.orbit_figure(points, circles, parts, n_times - 1)
    save_figure(figure, "invariant_decomposition")
    plt.close(figure)
    save_animation(render.orbit_animation(points, circles, parts, n_frames), "invariant_decomposition", frame_duration_ms)
    planes = core.placed_planes(motion)                                        # [3] Bivector+
    wedge_squares, wedge_parts = core.wedge_decomposition(bivector)            # [3] Scalar+, [3] Bivector+
    radii = circles.scalar_norm_squared()                                      # [3, n_times] Scalar+

    # checks
    # The planes sum to the bivector, commute, and are simple; the two decompositions agree on both squares and planes.
    assert (parts.sum(axis=0) - bivector).scalar_norm_squared() < 1e-22
    assert (parts[:, None].commutator(parts[None]).scalar_norm_squared() < 1e-22).all()
    assert ((parts ^ parts).scalar_norm_squared() < 1e-20).all()
    assert ((wedge_squares - squares).abs() < 1e-11).all()
    assert ((wedge_parts - parts).scalar_norm_squared() < 1e-22).all()
    # The factored rotor matches the exponential, preserves the direction's unit norm, and traces constant-radius circles in each plane.
    assert (core.factored_rotor(squares, parts, core.mv.scalar([1])) - (bivector / 2).exp()).scalar_norm_squared() < 1e-12
    assert ((points.scalar_norm_squared() - 1).abs() < 1e-6).all()
    assert ((radii - radii[:, :1]).abs() < 1e-6).all()
    # The placed planes of the projective motion stay simple, commute pairwise, and sum to the whole motion in six dimensions.
    np.testing.assert_allclose((planes ^ planes).kernel, 0.0, atol=1e-11)
    np.testing.assert_allclose(planes[:, None].commutator(planes[None]).kernel, 0.0, atol=1e-11)
    np.testing.assert_allclose((motion - planes.sum(axis=-1)).kernel, 0.0, atol=1e-10)


if __name__ == "__main__":
    main()
