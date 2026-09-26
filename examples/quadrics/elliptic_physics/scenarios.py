"""Scenes of quadric physics on S² and on S³, one engine instantiated for each sphere.

On S² the bodies are ellipses seen on the front hemisphere: a crowd of needles, discs and ovals; a
single oval tumbling about its intermediate axis; and a giant oval with an agile swarm.

On S³ each scene is chosen for what the sphere does to the view. The crowd: ellipsoids all over the
3-sphere, seen from a fixed eye. The gap: one huge ellipsoid leaves, between itself and its
own antipodal image, a belt around the great sphere orthogonal to its centre, thick where the
ellipsoid is short and thin where it is long; small ellipsoids bounce in it, and the eye sits in the belt
looking along it. The needle: an ellipsoid 80° long almost meets itself through the antipode of its
centre, and the eye looks at the 20° gap between its tips. The tunnel: the other real quadric of
the 3-sphere, the Clifford torus of the points at a fixed angle from the great circle
x == y == 0, is a wall that splits the sphere into two linked solid tori; ellipsoids bounce
inside one, and the eye on its core circle looks down the tube.
"""

from __future__ import annotations

import numpy as np

from numga.algebras import Spherical3D
from examples import instantiate
from examples.quadrics.elliptic_physics import render
from examples.quadrics.s3_raytracer import core as raytracer

S2 = instantiate("examples.quadrics.elliptic_physics.core", Spherical3D)
S3 = instantiate("examples.quadrics.elliptic_physics.core", raytracer.ga)


# --- S² -------------------------------------------------------------------------------
# The camera rotor, folded into the initial orientation of every body.
CAMERA = (S2.mv.yz * 0.2714904022294391 + S2.mv.zx * 0.6097774271693677 + S2.mv.xy * 1.0148400517968847).exp()


def ellipse_mesh(half_angles: np.ndarray, mass: np.ndarray, n_phi: int, n_r: int) -> tuple[S2.Point, S2.Scalar]:
    """A polar grid of mass points over each ellipse with half-angles (..., 2), in its gnomonic chart."""
    tx, ty = np.tan(half_angles[..., 0])[..., None, None], np.tan(half_angles[..., 1])[..., None, None]
    R, Phi = np.meshgrid(np.linspace(0.0, 1.0, n_r + 1)[1:], np.linspace(0.0, 2.0 * np.pi, n_phi + 1))
    x = tx * R * np.cos(Phi)
    y = ty * R * np.sin(Phi)
    # Sphere area element of the gnomonic parametrization, times trapezoid quadrature weights,
    # so the point masses approximate a uniform mass distribution over the ellipse.
    area = tx * ty * R / (1.0 + x**2 + y**2) ** 1.5
    # The omitted R == 0 node has zero area element.
    w_r = np.ones(n_r); w_r[-1] = 0.5
    w_phi = np.ones(n_phi + 1); w_phi[[0, -1]] = 0.5
    point_mass = (area * w_phi[:, None] * w_r[None, :]).reshape(area.shape[:-2] + (-1,))
    points = (S2.mv.yz * x + S2.mv.zx * y + S2.mv.xy).normalized().reshape(point_mass.shape)
    return points, S2.mv.scalar((point_mass * (mass[..., None] / point_mass.sum(axis=-1, keepdims=True)))[..., None])


def toward(theta: np.ndarray, phi: np.ndarray) -> S2.Motor:
    """Rotors carrying the pole to polar angle theta at azimuth phi."""
    # The plane zx turned about z by the azimuth.
    return (((S2.mv.xy * (-phi / 2)).exp() >> S2.mv.zx) * (theta / 2.0)).exp()


def ellipses(half_angles: np.ndarray, mass: np.ndarray, placement: S2.Motor, rate: np.ndarray, colors: list[str], n_phi: int) -> S2.Bodies:
    """Ellipses with half-angles (bodies, 2) in degrees, placed under the camera, spinning at body-frame
    rates (bodies, 3) on yz, zx and xy."""
    half_angles = np.radians(half_angles)
    rates = S2.mv.yz * rate[:, 0] + S2.mv.zx * rate[:, 1] + S2.mv.xy * rate[:, 2]
    return S2.body(render.rgb(colors), S2.ellipsoid(np.tan(half_angles)), CAMERA * placement, rates,
                   *ellipse_mesh(half_angles, mass, n_phi, 10))


def conserved(trajectory: S2.Trajectory, drift: float) -> None:
    """Checks: kinetic energy within the given relative drift, total momentum exactly."""
    energy, momentum = trajectory.energy.to_array(), trajectory.momentum.norm().to_array()
    assert np.ptp(energy) / abs(energy[0]) < drift
    assert np.ptp(momentum) / momentum[0] < 1e-11


def crowded(frames: int) -> tuple[S2.Trajectory, np.ndarray]:
    """Seven extreme shapes, needles to discs, scattered over S² and colliding."""
    bodies = ellipses(
        half_angles=np.array([[30.0, 6.0], [17.0, 17.0], [28.0, 6.5], [24.0, 5.5], [15.0, 4.0], [26.0, 9.0], [9.0, 9.0]]),
        mass=np.array([1.0, 1.2, 0.9, 0.8, 0.5, 1.1, 0.4]),
        placement=toward(np.array([0.35, 1.05, 1.15, 1.10, 1.25, 0.90, 1.55]), np.array([0.2, 0.7, 2.1, 3.6, 4.9, -0.67, 2.90])),
        rate=np.array([[0.8, 2.6, 0.5], [1.8, -1.2, 0.6], [-1.7, 1.5, -0.6], [2.0, 1.0, -0.5],
                       [-2.2, -1.8, 0.7], [1.6, 1.4, 0.7], [-2.4, 0.8, -0.4]]),
        colors=["#38bdf8", "#f43f5e", "#fbbf24", "#34d399", "#a855f7", "#fb923c", "#ec4899"],
        n_phi=384,
    )
    trajectory = S2.simulate(bodies, frames, 0.015, 6)

    # --- checks ---------------------------------------------------------------------------
    conserved(trajectory, 1e-3)
    return trajectory, bodies.color


def tumbling(frames: int) -> tuple[S2.Trajectory, np.ndarray]:
    """One oval 40° by 8° spinning near its intermediate axis: it flips over and back, periodically."""
    tilt = (S2.mv.xz * (np.radians(15.0) / 2.0)).exp() * (S2.mv.yz * (np.radians(10.0) / 2.0)).exp()
    bodies = ellipses(np.array([[40.0, 8.0]]), np.array([1.0]), tilt.reshape(1), np.array([[14.0, -0.1, 0.05]]), ["#38bdf8"], 384)
    trajectory = S2.simulate(bodies, frames, 0.015, 6)

    # --- checks ---------------------------------------------------------------------------
    conserved(trajectory, 5e-4)
    # The rate about the intermediate axis changes sign at least twice: it flips over and back,
    # the Dzhanibekov effect.
    spin = (trajectory.rate[:, 0] | S2.mv.yz).to_array()
    assert np.count_nonzero(np.diff(np.sign(spin))) >= 2
    return trajectory, bodies.color


def hyperbolic(frames: int) -> tuple[S2.Trajectory, np.ndarray]:
    """A giant oval, 75° by 48°, dominating the sphere, and a swarm of small bodies in the channel around it."""
    giant = ellipses(np.array([[75.0, 48.0]]), np.array([6.0]), toward(np.zeros(1), np.zeros(1)), np.array([[0.3, 0.2, 0.4]]), ["#38bdf8"], 1536)
    swarm = ellipses(
        half_angles=np.array([[18.0, 4.5], [10.0, 10.0], [15.0, 4.0], [16.0, 5.0]]),
        mass=np.array([0.6, 0.7, 0.5, 0.5]),
        placement=toward(np.full(4, 1.57), np.array([0.35, 1.10, 1.85, 2.65])),
        rate=np.array([[2.5, 1.5, -0.8], [-2.2, 1.2, 0.6], [1.8, -2.0, 0.7], [-1.6, -1.8, -0.5]]),
        colors=["#f43f5e", "#fbbf24", "#34d399", "#a855f7"],
        n_phi=384,
    )
    bodies = S2.Bodies.join(giant, swarm)
    trajectory = S2.simulate(bodies, frames, 0.015, 6)

    # --- checks ---------------------------------------------------------------------------
    conserved(trajectory, 5e-4)
    return trajectory, bodies.color


# --- S³ -------------------------------------------------------------------------------
def placed(place: S3.Point, spin: S3.Bivector) -> S3.Motor:
    """Motors carrying the origin to the given points, spun by the given rotation bivectors."""
    return (raytracer.unit(place) / raytracer.origin).square_root() * (spin * 0.5).exp()


def resting(color: np.ndarray, Q: S3.DualQuadric, rate: S3.Bivector, mass: float, rng: np.random.Generator) -> S3.Bodies:
    """One large body at the origin, unturned."""
    motor = placed(S3.mv.zyx.reshape(1), S3.mv.bivector(np.zeros((1, 6))))
    return S3.body(color[None], Q, motor, rate, *S3.filled(Q, np.array([mass]), 400, rng))


def population(rng: np.random.Generator, candidates: int, sizes: tuple[float, float]) -> S3.Bodies:
    """Random ellipsoids: half-widths log-uniform per axis between the sizes, so needles, discs and blobs;
    placed uniformly on the 3-sphere. Tennis-racket rates: spin about the intermediate axis, the
    middle half-width, whose bivector is the plane it is normal to (yz for the x axis, zx for y,
    xy for z), plus a nudge and a drift across the sphere."""
    half_widths = 10 ** rng.uniform(np.log10(sizes[0]), np.log10(sizes[1]), size=(candidates, 3))
    rate = rng.normal(size=(candidates, 6)) * np.array([0.3, 0.3, 0.3, 0.6, 0.6, 0.6])
    rate[np.arange(candidates), np.argsort(half_widths, axis=-1)[:, 1]] = rng.choice([-5.0, 5.0], size=candidates)
    place = S3.mv.antivector(rng.normal(size=(candidates, 4)))
    spin = S3.mv.bivector(rng.normal(size=(candidates, 6)) * np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]))
    color = render.hues(rng.permutation(candidates) / candidates)
    Q = S3.ellipsoid(half_widths)
    return S3.body(color, Q, placed(place, spin), S3.mv.bivector(rate), *S3.filled(Q, np.prod(half_widths, axis=-1) * 200.0, 400, rng))


def admitted(bodies: S3.Bodies, given: int, count: int) -> S3.Bodies:
    """The first `given` bodies plus the first `count` of the rest that overlap none kept before them:
    one batched overlap test over the pairs within reach, then a greedy pass over its boolean matrix."""
    i, j = S3.candidates_near(bodies)
    relative = bodies.motor[i].inverse() * bodies.motor[j]
    overlapping = np.zeros((bodies.color.shape[0],) * 2, dtype=bool)
    overlapping[i, j] = overlapping[j, i] = S3.overlap(bodies.C[i], relative >> bodies.C[j](relative << S3.Point))[0] < 0.0
    keep = list(range(given))
    for candidate in range(given, bodies.color.shape[0]):
        if not overlapping[candidate, keep].any():
            keep.append(candidate)
    assert len(keep) >= given + count, f"only {len(keep) - given} admissible non-overlapping candidates"
    return bodies.select(np.array(keep[:given + count]))


def collisions_conserve_energy(trajectory: S3.Trajectory) -> None:
    """Checks: bodies collided, and elastic impulses with a symplectic step kept the energy."""
    energy = trajectory.energy.to_array()
    assert trajectory.impulses > 0
    assert np.ptp(energy) / energy[0] < 2e-2


def crowd(frames: int):
    """28 ellipsoids all over the 3-sphere, seen from the origin."""
    rng = np.random.default_rng(3)
    bodies = admitted(population(rng, 120, (0.05, 0.5)), 0, 28)
    # The light: a point 0.8 rad from the eye, above and behind it, 70° off the line of sight so it
    # sits just outside the 120° frustum (the image corners reach 65°) in either direction.
    # With bodies all over the 3-sphere, whatever drifts behind the eye reappears ahead near the
    # antipode, which is what the sphere looks like.
    light = raytracer.unit(((S3.mv.xw * -0.34 + S3.mv.zw * 0.94) * (0.8 / 2)).exp() >> raytracer.origin)
    trajectory = S3.simulate(bodies, frames, 0.02, 8)

    # --- checks ---------------------------------------------------------------------------
    collisions_conserve_energy(trajectory)
    return trajectory, bodies.color, S3.mv.rotor(), light


def gap(frames: int):
    rng = np.random.default_rng(5)
    # The large object: an ellipsoid reaching 60°, 75° and 70° along x, y, z, so the belt around the great
    # sphere w == 0 between it and its antipodal image is 30°, 15° and 20° thick on either side.
    huge = resting(np.array([0.75, 0.7, 0.6]), S3.ellipsoid(np.tan(np.radians([[60.0, 75.0, 70.0]]))), S3.mv.bivector(np.zeros((1, 6))), 500.0, rng)
    bodies = admitted(S3.Bodies.join(huge, population(rng, 2000, (0.03, 0.15))), 1, 60)
    # The eye a quarter turn along x, in the middle of the belt's thick part, looking along y where
    # it thins; the light behind the eye and a little across the belt.
    eye = (S3.mv.xw * (np.pi / 2) * 0.5).exp() * (-S3.mv.xy * (np.pi / 2) * 0.5).exp()
    light = eye * ((S3.mv.xw * -0.3 + S3.mv.yw * 0.25 + S3.mv.zw * 0.2) * 0.5).exp() >> raytracer.origin
    trajectory = S3.simulate(bodies, frames, 0.02, 8)

    # --- checks ---------------------------------------------------------------------------
    collisions_conserve_energy(trajectory)
    return trajectory, bodies.color, eye, light


def needle(frames: int):
    rng = np.random.default_rng(7)
    # The large object: a heavy needle 80° long along x with a slightly oval cross-section, spinning
    # about its own axis, so its gap stays in the fixed view.
    long = resting(np.array([0.9, 0.85, 0.3]), S3.ellipsoid(np.tan(np.radians([[80.0, 4.0, 3.0]]))), S3.mv.yz.reshape(1) * 2.0, 50.0, rng)
    bodies = admitted(S3.Bodies.join(long, population(rng, 1000, (0.05, 0.3))), 1, 40)
    # The eye at the ideal point of x, where the tips almost meet, lifted 35° along z and looking
    # back down at the gap; the light behind and above the eye.
    eye = (S3.mv.xw * (np.pi / 2) * 0.5).exp() * (S3.mv.zw * np.radians(35.0) * 0.5).exp() * (S3.mv.zx * (-np.pi / 2) * 0.5).exp()
    light = eye * ((S3.mv.xw * -0.3 + S3.mv.yw * 0.2 + S3.mv.zw * 0.3) * 0.5).exp() >> raytracer.origin
    trajectory = S3.simulate(bodies, frames, 0.02, 8)

    # --- checks ---------------------------------------------------------------------------
    collisions_conserve_energy(trajectory)
    return trajectory, bodies.color, eye, light


def tunnel(frames: int):
    rng = np.random.default_rng(11)
    # The large object: a torus around the great circle x == y == 0, the dual quadric with -1 across
    # the tube in x and -1 / 1.5**2 in y, an elliptical cross-section, and 1 / np.tan(radius)**2
    # along it, with the tube's angular radius 20° at w and 40° a quarter turn along the core (z),
    # so the tube widens and narrows down the view.
    # Its inside, in the sense of its form, is the complementary solid torus; the crowd lives in the tube.
    tube = S3.quadric(np.array([[-1.0, -1.0 / 1.5**2, 1.0 / np.tan(np.radians(40.0))**2, 1.0 / np.tan(np.radians(20.0))**2]]))
    torus = resting(np.array([0.55, 0.65, 0.75]), tube, S3.mv.bivector(np.zeros((1, 6))), 500.0, rng)
    bodies = admitted(S3.Bodies.join(torus, population(rng, 2000, (0.03, 0.1))), 1, 50)
    # The eye in the wide section at +z, looking toward the narrow waist at -w; the light halfway to the wall, up and right.
    eye = (S3.mv.zw * (np.pi / 2) * 0.5).exp() * (S3.mv.zx * (np.pi / 2) * 0.5).exp()
    camera = eye >> raytracer.origin
    side_up = (S3.mv.zxw + S3.mv.xyw).normalized()
    wall_surface = torus.world()
    conic, polar = raytracer.project(eye, wall_surface)
    wall_hit = (camera * raytracer.reproject(conic, polar, side_up) + (eye >> side_up)).normalized()
    light = (camera + wall_hit).normalized()
    trajectory = S3.simulate(bodies, frames, 0.02, 8)

    # --- checks ---------------------------------------------------------------------------
    collisions_conserve_energy(trajectory)
    final = trajectory.surfaces[-1]
    # Nobody ended up in the wall.
    assert (S3.overlap(final[0], final[1:])[0] > -1e-2).all()
    np.testing.assert_allclose((wall_hit & wall_surface(wall_hit)).to_array(), 0.0, atol=1e-12)
    np.testing.assert_allclose((camera | light).to_array(), (light | wall_hit).to_array(), atol=1e-12)
    return trajectory, bodies.color, eye, light


if __name__ == "__main__":
    import argparse
    from functools import partial
    from examples.animation import save_animation, save_figure

    def on_s2(scene, name: str, frames: int, diagnostics) -> None:
        """Simulate a scene on S², save its diagnostic figure and its hemisphere animation."""
        trajectory, colors = scene(frames)
        save_figure(diagnostics(trajectory, colors, 0.015), name)
        save_animation(render.hemisphere_frames(trajectory.surfaces, colors, 240, 4), name, 15)

    def on_s3(scene, name: str, frames: int) -> None:
        """Simulate a scene on S³, trace it from its eye, save the last frame and the animation."""
        trajectory, colors, eye, light = scene(frames)
        images = render.s3_frames(trajectory, colors, eye, light, (180, 240), 4)
        save_figure(render.draw_last_frame(images, colors.shape[0]), name)
        save_animation(images, name, 50)

    SCENES = {
        "crowded": partial(on_s2, crowded, "spherical_quadric_physics", 160, render.draw_collisions),
        "tumbling": partial(on_s2, tumbling, "spherical_tumbling_oval", 240, render.draw_tumbling),
        "hyperbolic": partial(on_s2, hyperbolic, "spherical_hyperbolic_arena", 160, render.draw_collisions),
        "crowd": partial(on_s3, crowd, "s3_quadric_physics", 240),
        "gap": partial(on_s3, gap, "s3_gap", 240),
        "needle": partial(on_s3, needle, "s3_needle", 240),
        "tunnel": partial(on_s3, tunnel, "s3_tunnel", 240),
    }

    parser = argparse.ArgumentParser(description="Quadric physics on S² and S³")
    parser.add_argument("--scene", choices=[*SCENES, "all"], default="crowded")
    args = parser.parse_args()
    for scene in (SCENES if args.scene == "all" else [args.scene]):
        SCENES[scene]()
