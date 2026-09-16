"""Three scenes of quadric physics on the 3-sphere, each chosen for what the sphere does to the view.

The gap: one huge ellipsoidal cap leaves, between itself and its own antipodal image, a belt around
the great sphere orthogonal to its centre, thick where the cap is short and thin where it is long;
small caps bounce in it, and the eye sits in the belt looking along it. The needle: a cap 80° long
almost meets itself through the antipode of its centre, and the eye looks at the 20° gap between
its tips. The tunnel: the other real quadric of the 3-sphere, the Clifford torus x² + y² = tan²ρ
(z² + w²), is a wall that splits the sphere into two linked solid tori; caps bounce inside one,
and the eye on its core circle looks down the tube.
"""

from __future__ import annotations

import numpy as np

from examples import PLOT_DIR
from examples.animation import save_gif
from examples.sketches.s3_quadric_physics import body, cap_quadric, overlap, placed, population, quadric, run
from examples.sketches.spherical_raytracer import Plane, mv, origin, pixel_chart


def gap(animation_path: str = str(PLOT_DIR / "sketch_s3_gap.gif"), shape: tuple[int, int] = (180, 240), supersample: int = 4, frames: int = 240) -> None:
    rng = np.random.default_rng(5)
    # The large object: a cap reaching 60°, 75° and 70° along x, y, z, so the belt around the great
    # sphere w = 0 between it and its antipodal image is 30°, 15° and 20° thick on either side.
    huge = body(np.array([[0.75, 0.7, 0.6]]), cap_quadric(np.tan(np.radians([[60.0, 75.0, 70.0]]))), placed(mv.antivector(np.array([[0.0, 0.0, 0.0, 1.0]])), mv.bivector(np.zeros((1, 6)))), mv.bivector(np.zeros((1, 6))), np.array([500.0]), rng)
    bodies = population(rng, 60, 2000, (0.03, 0.15), huge)
    # The eye a quarter turn along x, in the middle of the belt's thick part, looking along y where
    # it thins; the light behind the eye and a little across the belt.
    eye = (mv.xw * (np.pi / 2) * 0.5).exp() * (-mv.xy * (np.pi / 2) * 0.5).exp()
    light = eye * ((mv.xw * -0.3 + mv.yw * 0.25 + mv.zw * 0.2) * 0.5).exp() >> origin
    chart = pixel_chart(np.radians(120.0), (shape[0] * supersample, shape[1] * supersample))
    frames_out, collisions, energies = run(bodies, lambda bodies: (eye, light), frames, chart, shape, supersample)
    if animation_path:
        save_gif(frames_out, animation_path, duration_ms=50)
    # --- checks: kernel-level assertions, deliberately outside the demonstration ----------
    assert collisions > 0
    assert np.ptp(energies) / energies[0] < 2e-2


def needle(animation_path: str = str(PLOT_DIR / "sketch_s3_needle.gif"), shape: tuple[int, int] = (180, 240), supersample: int = 4, frames: int = 240) -> None:
    rng = np.random.default_rng(7)
    # The large object: a heavy needle 80° long along x with a slightly oval cross-section, spinning
    # about its own axis, so its gap stays in the fixed view.
    long = body(np.array([[0.9, 0.85, 0.3]]), cap_quadric(np.tan(np.radians([[80.0, 4.0, 3.0]]))), placed(mv.antivector(np.array([[0.0, 0.0, 0.0, 1.0]])), mv.bivector(np.zeros((1, 6)))), mv.yz.reshape(1) * 2.0, np.array([50.0]), rng)
    bodies = population(rng, 40, 1000, (0.05, 0.3), long)
    # The eye at the ideal point of x, where the tips almost meet, lifted 35° along z and looking
    # back down at the gap; the light behind and above the eye.
    eye = (mv.xw * (np.pi / 2) * 0.5).exp() * (mv.zw * np.radians(35.0) * 0.5).exp() * (mv.zx * (-np.pi / 2) * 0.5).exp()
    light = eye * ((mv.xw * -0.3 + mv.yw * 0.2 + mv.zw * 0.3) * 0.5).exp() >> origin
    chart = pixel_chart(np.radians(120.0), (shape[0] * supersample, shape[1] * supersample))
    frames_out, collisions, energies = run(bodies, lambda bodies: (eye, light), frames, chart, shape, supersample)
    if animation_path:
        save_gif(frames_out, animation_path, duration_ms=50)
    # --- checks: kernel-level assertions, deliberately outside the demonstration ----------
    assert collisions > 0
    assert np.ptp(energies) / energies[0] < 2e-2


def tunnel(animation_path: str = str(PLOT_DIR / "sketch_s3_tunnel.gif"), shape: tuple[int, int] = (180, 240), supersample: int = 4, frames: int = 240) -> None:
    rng = np.random.default_rng(11)
    # The large object: a torus around the great circle x = y = 0, the dual quadric with -1 across
    # the tube in x and -1/1.5² in y, an elliptical cross-section, and 1/tan²ρ along it with ρ = 20°
    # at the eye (w) and 40° a quarter turn ahead (z), so the tube widens and narrows down the view.
    # Its inside, in the sense of its form, is the complementary solid torus; the crowd lives in the tube.
    tube = quadric(np.array([[-1.0, -1.0 / 1.5**2, 1.0 / np.tan(np.radians(40.0))**2, 1.0 / np.tan(np.radians(20.0))**2]]))
    torus = body(np.array([[0.55, 0.65, 0.75]]), tube, placed(mv.antivector(np.array([[0.0, 0.0, 0.0, 1.0]])), mv.bivector(np.zeros((1, 6)))), mv.bivector(np.zeros((1, 6))), np.array([500.0]), rng)
    bodies = population(rng, 50, 2000, (0.03, 0.1), torus)
    # The eye on the core circle at the origin looking along it, +z; the light behind the eye.
    eye = (mv.zx * (np.pi / 2) * 0.5).exp()
    light = eye * ((mv.xw * -0.3 + mv.yw * 0.15 + mv.zw * 0.15) * 0.5).exp() >> origin
    chart = pixel_chart(np.radians(120.0), (shape[0] * supersample, shape[1] * supersample))
    frames_out, collisions, energies = run(bodies, lambda bodies: (eye, light), frames, chart, shape, supersample)
    if animation_path:
        save_gif(frames_out, animation_path, duration_ms=50)
    # --- checks: kernel-level assertions, deliberately outside the demonstration ----------
    assert collisions > 0
    assert np.ptp(energies) / energies[0] < 2e-2
    world = bodies.motor >> bodies.Q(bodies.motor << Plane)
    assert (overlap(world.inverse()[0], world.inverse()[1:])[0] > -1e-2).all()   # nobody ended up in the wall


def main() -> None:
    gap()
    needle()
    tunnel()


if __name__ == "__main__":
    main()
