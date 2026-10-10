"""Unit tests for N-camera projective bundle adjustment with perspective cone quadrics."""

from __future__ import annotations

import signal
import matplotlib.pyplot as plt
import numpy as np
import pytest

from numga import stack
from numga.algebras import PGA2D, PGA3D
from examples import instantiate
from examples.estimation.multiview import render, scenarios
from examples.estimation.multiview.render import coordinates

core = instantiate("examples.estimation.multiview.core", PGA2D)
Point, mv, point = core.Point, core.mv, core.point


@pytest.fixture(autouse=True)
def ten_second_budget():
    """Every test in this module must finish within ten seconds."""
    def expired(signum, frame):
        raise TimeoutError("test exceeded the ten second budget")
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, 10.0)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def rmse(points: Point, truth: Point) -> float:
    """Root mean square distance between unit points: the difference is a direction, its length the norm of its complement."""
    return float((points - truth).dual().norm_squared().mean(axis=0).square_root().to_array())


def test_triangulate_cones():
    """Perspective cone pullback and fusion reconstruct landmarks given true poses."""
    rng = np.random.default_rng(123)
    xy = rng.uniform([-0.6, 1.0], [0.6, 2.5], size=(6, 2))
    true_points = point(xy)
    motors = scenarios.rig(np.array([-0.75, 0.75]), np.radians([18.0, -18.0]))

    camera = (point(np.zeros(2)) & Point) ^ (mv.y - mv.w)
    cameras = camera.broadcast_to((2,))
    projs = cameras(motors << true_points[:, None])
    pixels = projs / (mv.w & projs)

    # A sensor quadric at the principal point, moved to each pixel, pulled back into cones:
    q_sensor = mv.x * (mv.x & Point)
    sensor_discs = core.sensor_disk_at(pixels, point(np.array([0.0, 1.0])), q_sensor)
    local_cones = core.make_cones(cameras, sensor_discs)
    points, fused = core.triangulate_cones(motors, local_cones)
    np.testing.assert_allclose(coordinates(points), xy, atol=1e-5)

    # The fused precision is positive on every displacement, so the splat is a bounded ellipse:
    angles = np.linspace(0.0, np.pi, 12, endpoint=False)
    displacements = mv("yw wx", np.stack([np.cos(angles), np.sin(angles)], axis=-1))
    precision = (fused[:, None](displacements[None, :]) & displacements[None, :]).to_array()
    assert np.all(precision > 0)


def test_bundle_adjust_converges():
    """Alternating Newton steps on the cone value recover the scene from a perturbed camera."""
    true_motors = scenarios.three_camera_truth()
    true_points, _, local_cones = scenarios.observe(true_motors)
    initial_motors = scenarios.rig(np.array([-0.75, 0.75, 0.0]), np.radians([18.0, -18.0 * 1.05, 0.0]))
    initial_points, _ = core.triangulate_cones(initial_motors, local_cones)

    _, points, _ = core.bundle_adjust(initial_motors, local_cones, 12, 0.9, np.array([0.0, 1.0, 0.0]))
    assert rmse(points, true_points) < rmse(initial_points, true_points)
    assert rmse(points, true_points) < 0.02


def test_schur_bundle_adjust_converges_and_yields_information():
    """The joint Newton step with the Schur complement converges and returns pose information forms."""
    true_motors = scenarios.three_camera_truth()
    true_points, cameras, local_cones = scenarios.observe(true_motors)
    initial_motors = scenarios.rig(np.array([-0.75, 0.75, 0.0]), np.radians([18.0, -18.0 * 1.05, 0.0]))
    initial_points, _ = core.triangulate_cones(initial_motors, local_cones)

    _, points, _, information = core.bundle_adjust_schur(
        cameras, initial_motors, local_cones, 10, 0.7, np.array([0.0, 1.0, 0.0]),
    )
    assert rmse(points, true_points) < rmse(initial_points, true_points)
    assert rmse(points, true_points) < 0.02

    # The information on all poses together is a symmetric form over the field of camera twists, zero
    # on the anchored cameras and positive on the free one:
    assert information.gatype.site_shape == (3, 3)
    basis = mv("yw wx xy", np.eye(3))
    units = (basis[None, :, None] * np.eye(3)[:, None, :]).field()          # [cams, 3] Twist[cams]
    gram = information(units[:, :, None, None], units[None, None]).to_array()   # [cams, 3, cams, 3]
    np.testing.assert_allclose(gram, gram.transpose(2, 3, 0, 1), atol=1e-9)
    np.testing.assert_allclose(gram[[0, 2]], 0.0)
    np.testing.assert_allclose(gram[:, :, [0, 2]], 0.0)
    assert np.linalg.eigvalsh(gram[1, :, 1]).min() > 0


def test_the_schur_step_moves_two_free_cameras_together():
    """With two cameras free, a point's response to a step of one changes the cost seen by the other: the
    step that accounts for it settles within a few iterations. With one camera anchored the scale of the
    scene is free, so the cone cost, not the distance to the true poses, measures convergence."""
    true_motors = scenarios.three_camera_truth()
    _, cameras, local_cones = scenarios.observe(true_motors)
    motors = scenarios.rig(np.array([-scenarios.BASELINE, scenarios.BASELINE * 1.1, 0.15]),
                           np.array([scenarios.GAZE, -scenarios.GAZE * 1.25, 0.12]))
    free = np.array([0.0, 1.0, 1.0])

    def cost(motors):
        points, _ = core.triangulate_cones(motors, local_cones)
        world_cones = motors >> local_cones(motors << Point)
        return float((world_cones(points[:, None]) & points[:, None]).sum().to_array())

    initial = cost(motors)
    for _ in range(5):
        previous, motors = motors, core.bundle_adjust_schur(cameras, motors, local_cones, 1, 1.0, free)[0]
    assert cost(motors) < 1e-5 * initial
    assert np.abs((motors - previous).kernel).max() < 1e-5


def test_splat_alignment_converges_with_the_schur_step():
    """Aligning sight lines to the other cameras' beliefs, weighted into pixel units by depths read off
    those beliefs, recovers the perturbed camera as the Schur step does, without extracting a point."""
    true_motors = scenarios.three_camera_truth()
    true_points, cameras, local_cones = scenarios.observe(true_motors)
    initial_motors = scenarios.rig(np.array([-0.75, 0.75, 0.0]), np.radians([18.0, -18.0 * 1.05, 0.0]))
    free = np.array([0.0, 1.0, 0.0])
    projs = cameras(true_motors << true_points[:, None])
    pixels = projs / (mv.w & projs)

    splat_motors, _ = core.align_rays_to_splats(initial_motors, local_cones, point(np.zeros(2)), pixels, 4, 1.0, free)
    schur_motors, _, _, _ = core.bundle_adjust_schur(cameras, initial_motors, local_cones, 10, 1.0, free)
    np.testing.assert_allclose((splat_motors - true_motors).kernel, 0.0, atol=1e-10)
    np.testing.assert_allclose((splat_motors - schur_motors).kernel, 0.0, atol=1e-10)


# --- the same module in 3D --------------------------------------------------------------
core3 = instantiate("examples.estimation.multiview.core", PGA3D)


def coordinates_3d(points) -> np.ndarray:
    k = points.kernel
    return k[..., :3] / k[..., 3:]


def rig_3d(instance) -> tuple:
    """Three convergent cameras and eight landmarks in PGA3D, with their sight cones, in the context
    of the given instance of the core.

    Cam 0 left, panned right; Cam 1 right, panned left; Cam 2 central, raised and looking
    slightly down. The landmarks span depths z in [1.2, 2.65].
    """
    mv3 = instance.mv
    theta = np.radians(18.0)
    true_motors = stack([
        (-mv3.xw * 0.75 / 2).exp() * (-mv3.zx * theta / 2).exp(),
        (mv3.xw * 0.75 / 2).exp() * (mv3.zx * theta / 2).exp(),
        (mv3.yw * 0.35 / 2).exp() * (-mv3.yz * np.radians(12.0) / 2).exp(),
    ])
    camera = (instance.point(np.zeros(3)) & instance.Point) ^ (mv3.z - mv3.w)
    cameras = camera.broadcast_to((3,))
    xyz = np.array([
        [ 0.15, -0.20, 1.20],
        [-0.43,  0.15, 1.45],
        [ 0.50, -0.10, 1.70],
        [ 0.03,  0.25, 1.95],
        [ 0.65, -0.15, 2.30],
        [-0.60,  0.10, 2.65],
        [ 0.20,  0.30, 2.10],
        [-0.25, -0.25, 1.60],
    ])
    projs = cameras(true_motors << instance.point(xyz)[:, None])
    pixels = projs / (mv3.w & projs)
    # 2D transverse uncertainty on the sensor plane (z = 1) around the principal point:
    q_sensor = mv3.x * (mv3.x & instance.Point) + mv3.y * (mv3.y & instance.Point)
    sensor_discs = instance.sensor_disk_at(pixels, instance.point(np.array([0.0, 0.0, 1.0])), q_sensor)
    return true_motors, xyz, instance.make_cones(cameras, sensor_discs)


@pytest.mark.parametrize("rotation_deg, translation, iterations", [
    ((30.0, -20.0, 25.0), (-0.40, 0.30, 0.45), 30),
    ((50.0, 35.0, -40.0), (0.60, -0.50, 0.80), 40),
])
def test_3d_bundle_adjust_recovers_a_badly_perturbed_camera(rotation_deg, translation, iterations):
    """In PGA3D, the moving camera returns from pose errors of tens of degrees and most of a metre."""
    mv3 = core3.mv
    true_motors, xyz, local_cones = rig_3d(core3)
    rx, ry, rz = np.radians(rotation_deg)
    tx, ty, tz = translation
    perturbation = ((mv3.xw * tx + mv3.yw * ty + mv3.zw * tz) / 2).exp() * ((mv3.yz * rx + mv3.zx * ry + mv3.xy * rz) / 2).exp()
    motors = stack([true_motors[0], perturbation * true_motors[1], true_motors[2]])

    def cost(motors, points):
        local_points = motors << points[:, None]
        return float((local_cones(local_points) & local_points).sum().to_array())

    initial_points, _ = core3.triangulate_cones(motors, local_cones)
    est_motors, est_points, _ = core3.bundle_adjust(motors, local_cones, iterations, 0.7, np.array([0.0, 1.0, 0.0]))

    origin = core3.point(np.zeros(3))
    np.testing.assert_allclose(coordinates_3d(est_motors[1] >> origin), [0.75, 0.0, 0.0], atol=0.02)
    np.testing.assert_allclose(coordinates_3d(est_points), xyz, atol=0.02)
    # The residual rotation's scalar part is the cosine of half its angle:
    residual = (est_motors[1] * true_motors[1].reverse()).cast(PGA3D.subspace("1")).kernel[0]
    assert np.degrees(2.0 * np.arccos(min(abs(residual), 1.0))) < 0.5
    assert cost(est_motors, est_points) < cost(motors, initial_points) * 1e-3


def test_3d_bundle_adjust_converges_from_a_small_perturbation():
    """In PGA3D, the second camera panned 5% too far aligns with the other two anchored."""
    mv3 = core3.mv
    theta = np.radians(18.0)
    true_motors, xyz, local_cones = rig_3d(core3)
    motors = stack([
        true_motors[0],
        (mv3.xw * 0.75 / 2).exp() * (mv3.zx * theta * 1.05 / 2).exp(),
        true_motors[2],
    ])
    truth = core3.point(xyz)
    initial_points, _ = core3.triangulate_cones(motors, local_cones)
    _, points, _ = core3.bundle_adjust(motors, local_cones, 10, 0.9, np.array([0.0, 1.0, 0.0]))
    error = lambda p: float((p - truth).dual().norm_squared().mean(axis=0).square_root().to_array())
    assert error(points) < error(initial_points)
    assert error(points) < 0.02


def test_3d_coupled_newton_step_over_a_field_of_free_cameras():
    """The second derivative of the point-free misfit with respect to the twists of the free cameras,
    taken as a field, is the coupled Hessian, every camera against every other; solving it against the
    gradient gives the Newton step that the Hessian of all twist coefficients at once gives."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from numga.backend.jax import JaxContext, derivative

    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        jax_core = instantiate("examples.estimation.multiview.core", PGA3D, JaxContext)
        mv3, Point3, Plane3, Twist3 = jax_core.mv, jax_core.Point, jax_core.Plane, jax_core.Twist
        true_motors, _, local_cones = rig_3d(jax_core)
        perturb = ((mv3.xw * 0.05 - mv3.yw * 0.03 + mv3.zw * 0.04) / 2).exp() * ((mv3.yz * 0.04 - mv3.zx * 0.03 + mv3.xy * 0.05) / 2).exp()
        anchor = true_motors[0]                                              # Motor
        free = stack([perturb * true_motors[1], perturb.reverse() * true_motors[2]]).field()   # Motor[free]
        anchored = anchor >> local_cones[:, 0](anchor << Point3)             # [points] Plane <- Point
        moving = local_cones[:, 1:].field()                                  # [points] Plane[free] <- Point
        w = mv3.w

        def misfit(twists):
            """The fused cones' value at their own vertices, summed over the points: the pole of the plane
            at infinity under each fused cone is its vertex."""
            motors = free * (twists * 0.5).exp()                             # Motor[free]
            fused = anchored + (motors >> moving(motors << Point3)).sites.sum()
            pole = fused.dual().outermorphism(Plane3)(w).dual_inverse()      # [points] Point
            return ((pole & fused(pole)) / (w & pole) ** 2).sum()

        @jax.jit
        def newton(twists):
            gradient = derivative(misfit)(twists)                            # Scalar <- Twist[free]
            hessian = derivative(derivative(misfit))(twists)                 # Scalar <- (Twist[free], Twist[free])
            return hessian, -hessian.solve(gradient)                         # Twist[free]

        @jax.jit
        def flat_newton(coefficients):
            flat = lambda coefficients: misfit(jax_core.ctx.extensor(Twist3, coefficients.reshape(2, 6)).field()).kernel[0]
            return -jnp.linalg.solve(jax.hessian(flat)(coefficients), jax.grad(flat)(coefficients))

        hessian, step = newton(mv3.bivector(np.zeros((2, 6))).field())
        reference = flat_newton(jnp.zeros(12))
        np.testing.assert_allclose(step.kernel, reference.reshape(2, 6), atol=1e-10)
        # The cameras are coupled through the points they share:
        assert np.abs(hessian.kernel[0, 1]).max() > 0.1 * np.abs(hessian.kernel[0, 0]).max()
    finally:
        jax.config.update("jax_enable_x64", previous)


# --- scenarios through render -------------------------------------------------------------
def test_figure_renders():
    """The figure scenario passes its checks, and renders."""
    figure = render.draw_reconstruction(*scenarios.bundle_adjustment())
    assert isinstance(figure, plt.Figure)


@pytest.mark.parametrize("name", ["one_camera", "two_cameras", "three_cameras"])
def test_convergence_animations_render(name):
    """Each alternating convergence scenario renders a short animation."""
    frames = render.animate_convergence(getattr(scenarios, name)(1))
    assert frames and frames[0].ndim == 3 and all(f.shape == frames[0].shape for f in frames)


def test_schur_animation_renders():
    """The Schur convergence scenario renders a short animation with pose covariances."""
    frames = render.animate_convergence_with_covariance(scenarios.schur(1))
    assert frames and frames[0].ndim == 3 and all(f.shape == frames[0].shape for f in frames)
