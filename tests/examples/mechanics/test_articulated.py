"""The humanoid's dynamics: inverse and forward dynamics undo each other, the mass matrix is inverse dynamics' columns
and symmetric with the tree's sparsity, the energy falls in a fall, a batch runs as its humanoids do alone, a free
body's momentum turns as Euler's equations say, and the animation renders."""

from __future__ import annotations

from itertools import islice

import numpy as np

from examples.mechanics.articulated import core, render, scenarios
from examples.mechanics.articulated.core import Rate, mv

n_joints = len(scenarios.joint_bodies)
n_links = 1 + n_joints
rng = np.random.default_rng(3)
tree, robot = scenarios.robot()
# What acts on the humanoid dropped: its weights and hinges, the ground and the stops.
physics, ground, stops = scenarios.physics(), scenarios.ground(), scenarios.stops()
loads = scenarios.drop_loads(tree, robot)


def random_joints(n_robots: int) -> core.Scalar:
    """Joint values `[robots] Scalar[joints]`, drawn from a normal distribution."""
    return mv.scalar(rng.normal(size=(n_robots, n_joints, 1))).field()


def random_placement(n_robots: int) -> core.Motor:
    """Each robot's links placed at random joint angles, its torso dropped, tilted and spun at random,
    `[robots] Motor[links]`."""
    state = scenarios.drop(rng.uniform(1.0, 1.5, n_robots), rng.normal(size=n_robots), rng.normal(size=n_robots))
    return core.forward_kinematics(tree, robot, state.motor, random_joints(n_robots))


def largest(difference) -> float:
    """The largest coefficient of a difference."""
    return float(np.abs(np.asarray(difference.kernel)).max())


def test_forward_dynamics_undoes_inverse_dynamics():
    motors = random_placement(5)
    rate, acceleration = (mv.bivector(rng.normal(size=(5, 6))) for _ in range(2))
    speeds, accelerations = random_joints(5), random_joints(5)
    # Any forques on the links from outside.
    external = mv.antibivector(rng.normal(size=(5, n_links, 6))).field()        # [robots] Forque-[links]
    forque, forces = core.inverse_dynamics(tree, robot, motors, rate, speeds, acceleration, accelerations, external)
    # The body's forque is one more load on the body's link.
    loaded = external + ~tree.body * forque[..., None].field()
    recovered = core.forward_dynamics(tree, robot, motors, rate, speeds, forces, loaded)
    assert largest(recovered[0] - acceleration) < 1e-8 and largest(recovered[1] - accelerations) < 1e-8


def test_mass_matrix_is_inverse_dynamics_columns():
    """At rest and with nothing acting from outside, inverse dynamics of an acceleration is the mass matrix applied to
    it, block by block; the joints' block is symmetric and couples two joints only where one lies on the other's
    path."""
    motors = random_placement(1)
    whole, coupling, joints = core.mass_matrix(tree, robot, motors)
    units = mv.scalar(np.eye(n_joints)[:, None, :, None]).field()             # [joints, 1] Scalar+[joints]
    still = mv.bivector(np.zeros(6))
    external = mv.antibivector(np.zeros((n_links, 6))).field()               # Forque-[links]
    forque, forces = core.inverse_dynamics(tree, robot, motors, still, units * 0, still, units, external)
    assert largest(forque - coupling.adjugate()(units)) < 1e-10 and largest(forces - joints(units)) < 1e-10
    acceleration = mv.bivector(rng.normal(size=6))
    forque, forces = core.inverse_dynamics(tree, robot, motors, still, units[0] * 0, acceleration, units[0] * 0,
                                           external)
    assert largest(forque - whole(acceleration)) < 1e-10 and largest(forces - coupling(acceleration)) < 1e-10
    entries = (joints(units)[:, None] & units[None, :]).sites.sum().to_array()   # [joints, joints, 1]
    np.testing.assert_allclose(entries, np.swapaxes(entries, 0, 1), atol=1e-12)
    on_paths = np.zeros((n_links, n_joints))
    on_paths[tree.paths.rows, tree.paths.columns] = 1
    assert len(joints.rows) == np.count_nonzero(on_paths.T @ on_paths) < n_joints**2


def test_energy_falls_in_a_fall():
    """The implicit step takes energy away, but for what a part touching down within a step stores: it meets the
    ground only at the step's end, its spring pressed in unresisted. Over a fall the energy drops to a small share, and
    no frame gains more than a percent or two of it."""
    state = scenarios.drop(np.array([scenarios.drop_height, 1.3, 1.8]), np.array([scenarios.drop_tilt, -0.4, 0.6]),
                           np.array([scenarios.drop_spin, -1.0, 0.0]))
    energies = [core.energy(tree, robot, physics, ground, stops, state)]
    for _ in range(scenarios.n_frames):
        state = core.advance(tree, robot, state, scenarios.dt, scenarios.n_steps, loads)
        energies.append(core.energy(tree, robot, physics, ground, stops, state))
    rises = [((after - before) / energies[0].abs()).to_array().max() for before, after in zip(energies, energies[1:])]
    assert max(rises) < 0.03
    assert ((energies[-1] / energies[0]).to_array() < 0.25).all()


def test_batch_runs_as_each_robot_alone():
    heights, tilts, spins = np.array([1.45, 1.3, 1.6]), np.array([0.15, -0.1, 0.0]), np.array([1.0, 0.0, -1.0])
    n_steps = 40
    together = core.advance(tree, robot, scenarios.drop(heights, tilts, spins), scenarios.dt, n_steps, loads)
    for index in range(len(heights)):
        alone = core.advance(tree, robot, scenarios.drop(heights[index:index + 1], tilts[index:index + 1],
                                                                  spins[index:index + 1]),
                             scenarios.dt, n_steps, loads)
        assert all(largest(part[index] - single[0]) < 2e-10 for part, single in zip(together, alone))


def test_free_body_turns_as_euler_says():
    """A free body's momentum keeps still in space, so that `inertia(acceleration) == -rate.commutator(momentum)`:
    about its centre and in its principal frame, Euler's equations for a solid ellipsoid."""
    semi = np.array([0.3, 0.2, 0.1])
    mass = 2.0
    # A solid ellipsoid's six samples reach the square root of three fifths of each semi-axis.
    samples = mv.point(scenarios.principal_samples(np.sqrt(3 / 5) * semi))   # [6] Point-
    inertia = samples.regressive(samples.commutator(Rate) * (mass / 6)).sum(axis=-1)
    # A turn about the centre at an angular velocity: `mv.yz` turns points about x the negative way, as `mv.zx` does
    # about y and `mv.xy` about z.
    omega = np.array([0.7, -1.1, 0.4])
    rate = -(mv.yz * omega[0] + mv.zx * omega[1] + mv.xy * omega[2])
    acceleration = inertia.solve(-rate.commutator(inertia(rate)))
    moments = mass * (np.sum(semi**2) - semi**2) / 5
    alpha = (np.roll(moments, -1) - np.roll(moments, -2)) * np.roll(omega, -1) * np.roll(omega, -2) / moments
    expected = -(mv.yz * alpha[0] + mv.zx * alpha[1] + mv.xy * alpha[2])
    assert largest(acceleration - expected) < 1e-12


def test_animation_renders():
    _, _, _, _, radii = scenarios.home_pose()
    centres = scenarios.fall(scenarios.dropped())
    frames = list(islice(render.animate(centres, radii, len(scenarios.capsule_bodies)), 2))
    assert frames[0].shape[1:] == (render.n_pixels, 3) and frames[0].shape == frames[1].shape
