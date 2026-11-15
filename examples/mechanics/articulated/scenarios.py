"""Scene parameters and benchmarks define Brax's seventeen-hinge humanoid falling onto level ground and flying free across batched JAX compilations."""

from __future__ import annotations

from functools import partial

import numpy as np

from numga import stack
from numga.sparse import Sparse
from numga.topology import Topology
from examples.mechanics.articulated import core
from examples.mechanics.articulated.core import (
    Hinge,
    Inertia,
    Motor,
    Plane,
    Point,
    Quadric,
    Rate,
    Scalar,
    State,
    Tree,
    mv,
)

# Parent body index and home-pose offset in metres for the torso, waist, pelvis, legs, and arms.
body_parents = np.array([-1, 0, 1, 2, 3, 4, 2, 6, 7, 0, 9, 0, 11])
body_offsets = np.array([
    [0, 0, 0],
    [-0.01, 0, -0.26],
    [0, 0, -0.165],
    [0, -0.1, -0.04],
    [0, 0.01, -0.403],
    [0, 0, -0.45],
    [0, 0.1, -0.04],
    [0, -0.01, -0.403],
    [0, 0, -0.45],
    [0, -0.17, 0.06],
    [0.18, -0.18, -0.18],
    [0, 0.17, 0.06],
    [0.18, 0.18, -0.18],
])
# Hinge carrier body, pivot offset, axis direction, passive stiffness and damping, and stop bounds in degrees.
joint_bodies = np.array([1, 1, 2, 3, 3, 3, 4, 6, 6, 6, 7, 9, 9, 10, 11, 11, 12])
joint_pivots = np.array(
    [[0, 0, 0.065], [0, 0, 0.065], [0, 0, 0.1]]
    + [[0, 0, 0]] * 3
    + [[0, 0, 0.02]]
    + [[0, 0, 0]] * 3
    + [[0, 0, 0.02]]
    + [[0, 0, 0]] * 6
)
joint_directions = np.array([
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, -1, 0],
    [-1, 0, 0],
    [0, 0, -1],
    [0, 1, 0],
    [0, -1, 0],
    [2, 1, 1],
    [0, -1, 1],
    [0, -1, 1],
    [2, -1, 1],
    [0, 1, 1],
    [0, -1, -1],
])
joint_stiffness = np.array([20, 10, 10, 10, 10, 20, 0, 10, 10, 20, 1, 1, 1, 0, 1, 1, 0])
joint_damping = np.array([5, 5, 5, 5, 5, 5, 1, 5, 5, 5, 1, 1, 1, 1, 1, 1, 1])
joint_ranges = np.array([
    [-45, 45],
    [-75, 30],
    [-35, 35],
    [-25, 5],
    [-60, 35],
    [-110, 20],
    [-160, -2],
    [-25, 5],
    [-60, 35],
    [-110, 20],
    [-160, -2],
    [-85, 60],
    [-85, 60],
    [-90, 50],
    [-60, 85],
    [-60, 85],
    [-90, 50],
])
# Solid capsules and spheres at MuJoCo's default water density (1000 kg / m^3).
capsule_bodies = np.array([0, 0, 1, 2, 3, 4, 6, 7, 9, 10, 11, 12])
capsule_starts = np.array([
    [0, -0.07, 0],
    [-0.01, -0.06, -0.12],
    [0, -0.06, 0],
    [-0.02, -0.07, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0.01, 0.01, 0.01],
    [0, 0, 0],
    [0.01, -0.01, 0.01],
])
capsule_ends = np.array([
    [0, 0.07, 0],
    [-0.01, 0.06, -0.12],
    [0, 0.06, 0],
    [-0.02, 0.07, 0],
    [0, 0.01, -0.34],
    [0, 0, -0.3],
    [0, -0.01, -0.34],
    [0, 0, -0.3],
    [0.16, -0.16, -0.16],
    [0.17, 0.17, 0.17],
    [0.16, 0.16, -0.16],
    [0.17, -0.17, 0.17],
])
capsule_radii = np.array([0.07, 0.06, 0.06, 0.09, 0.06, 0.049, 0.06, 0.049, 0.04, 0.031, 0.04, 0.031])
sphere_bodies = np.array([0, 5, 8, 10, 12])
sphere_centres = np.array([[0, 0, 0.19], [0, 0, 0.1], [0, 0, 0.1], [0.18, 0.18, 0.18], [0.18, -0.18, 0.18]])
sphere_radii = np.array([0.09, 0.075, 0.075, 0.04, 0.04])
density = 1000.0
limit_stiffness = 2500.0
gravity_strength = 9.81
contact_stiffness_value, contact_damping, friction = 3e4, 3.0, 30.0
drop_height, drop_tilt, drop_spin = 1.45, 0.15, 1.0
dt = 0.01
n_steps = 4
n_frames = 60
frame_duration_ms = 40
joint_names = [
    "abdomen_z",
    "abdomen_y",
    "abdomen_x",
    "right_hip_x",
    "right_hip_z",
    "right_hip_y",
    "right_knee",
    "left_hip_x",
    "left_hip_z",
    "left_hip_y",
    "left_knee",
    "right_shoulder1",
    "right_shoulder2",
    "right_elbow",
    "left_shoulder1",
    "left_shoulder2",
    "left_elbow",
]
flight_height, flight_speed, flight_seed = 1.4, 0.5, 7
robot_counts = 4 ** np.arange(7)
n_timed = 5
# Intrinsic unit scalar reorienting an intrinsic point join into the extrinsic line through those points.
flip = mv.antipseudoscalar([1.0])                                              # [] Antipseudoscalar-
origin = mv.w.dual()                                                           # [] Point-


# --- math -----------------------------------------------------------------------------
def humanoid() -> tuple[Topology, np.ndarray, np.ndarray, int]:
    """Directed link `Topology`, body-carrier link indices, home-pose body origins, and tree generation count assemble from the humanoid's body and hinge tables."""
    n_bodies = len(body_parents)
    origins, carriers, depths, edges = np.zeros((n_bodies, 3)), np.zeros(n_bodies, dtype=int), [0], []
    for body_index, parent_index in enumerate(body_parents[1:], start=1):
        origins[body_index] = origins[parent_index] + body_offsets[body_index]
        link = carriers[parent_index]
        for _ in range(np.count_nonzero(joint_bodies == body_index)):
            edges.append([link, len(depths)])
            depths.append(depths[link] + 1)
            link = len(depths) - 1
        carriers[body_index] = link
    return Topology(len(depths), np.array(edges)), carriers, origins, max(depths)


def principal_samples(reaches: np.ndarray) -> np.ndarray:
    """Six symmetric sample offsets (`[..., 6, 3]`) along the coordinate axes match the principal second moments of a part with axial half-reaches `reaches` (`[..., 3]`)."""
    offsets = reaches[..., None, :] * np.eye(3)                               # [..., 3, 3]
    return np.concatenate([offsets, -offsets], axis=-2)


def home_pose() -> tuple[Rate, Point, np.ndarray, Point, np.ndarray]:
    """Home-pose hinge axes (`Rate+[E]`), six-point mass samples (`[n_parts, 6] Point-`) and sample masses (`[n_parts, 1]`), and contact ball centres (`[n_balls] Point-`) and radii (`[n_balls]`) describe the standing humanoid."""
    _, _, origins, _ = humanoid()
    n_joints = len(joint_bodies)
    n_capsules = len(capsule_bodies)
    n_parts = n_capsules + len(sphere_bodies)
    n_balls = 2 * n_capsules + len(sphere_bodies)
    planes = stack((mv.x, mv.y, mv.z))                                         # [3] Plane+
    pivots = mv.point(origins[joint_bodies] + joint_pivots)                    # [n_joints] Point-
    directions = mv.direction(joint_directions / np.linalg.norm(joint_directions, axis=-1, keepdims=True))
    axes = ((directions & pivots) * flip).field()                              # Rate+[E]
    part_bodies = np.concatenate([capsule_bodies, sphere_bodies])
    starts = origins[part_bodies] + np.concatenate([capsule_starts, sphere_centres])
    ends = origins[part_bodies] + np.concatenate([capsule_ends, sphere_centres])
    radii = np.concatenate([capsule_radii, sphere_radii])
    lengths = np.linalg.norm(ends - starts, axis=-1)
    cylinder_masses = density * np.pi * radii**2 * lengths
    ball_masses = density * 4 / 3 * np.pi * radii**3
    masses = cylinder_masses + ball_masses
    axial_moments = cylinder_masses * radii**2 / 2 + ball_masses * 2 * radii**2 / 5
    transverse_moments = (
        cylinder_masses * (radii**2 / 4 + lengths**2 / 12)
        + ball_masses * (2 * radii**2 / 5 + lengths**2 / 4 + 3 * lengths * radii / 8)
    )
    reaches = np.sqrt(
        3 * np.stack([axial_moments / 2, axial_moments / 2, transverse_moments - axial_moments / 2], axis=-1) / masses[:, None]
    )
    upward_axes = np.where((ends - starts)[:, 2:] < 0, starts - ends, ends - starts)
    normals = np.where(lengths[:, None] > 0, upward_axes, [[0, 0, 1]])
    facing_planes = (planes * normals).sum(axis=-1).normalized()               # [n_parts] Plane+
    part_turns = (1 + facing_planes * mv.z).normalized()                       # [n_parts] Motor+
    part_shifts = ((mv.w ^ (mv.point((starts + ends) / 2) - origin).dual()) / 2).exp()   # [n_parts] Motor+
    part_motors = (part_shifts * part_turns)[:, None]                          # [n_parts, 1] Motor+
    samples = part_motors >> mv.point(principal_samples(reaches))              # [n_parts, 6] Point-
    centres = mv.point(np.concatenate([starts[:n_capsules], ends[:n_capsules], starts[n_capsules:]]))   # [n_balls] Point-
    return axes, samples, masses[:, None] / 6, centres, np.concatenate([radii[:n_capsules], radii])


def humanoid_tree() -> tuple[Tree, Sparse]:
    """Kinematic `Tree` and part-carrier reading (`Scalar+[n_parts <- V]`) wire the humanoid's capsules, spheres, and contact balls to its links."""
    topology, carriers, _, n_generations = humanoid()
    ball_carriers = carriers[np.concatenate([capsule_bodies, capsule_bodies, sphere_bodies])]
    part_carriers = carriers[np.concatenate([capsule_bodies, sphere_bodies])]
    return core.kinematic_tree(topology, ball_carriers, n_generations), topology.reading(part_carriers, 0)


def humanoid_model() -> tuple[Tree, Rate, Inertia, Point]:
    """Kinematic `Tree`, home-pose hinge `axes` (`Rate+[E]`), link `inertias` (`(Forque- <- Rate+)[V]`), and weighted link `mass_points` (`Point-[V]`) define the standing humanoid."""
    tree, part_reading = humanoid_tree()
    axes, samples, masses, _, _ = home_pose()
    n_parts = len(samples)
    part_inertias = (masses * samples).regressive(samples.commutator(Rate)).sum(axis=-1)   # [n_parts] Forque- <- Rate+
    inertias = part_reading.adjugate() * part_inertias.field()                             # (Forque- <- Rate+)[V]
    mass_points = part_reading.adjugate() * (masses * samples).sum(axis=-1).field()        # Point-[V]
    return tree, axes, inertias, mass_points


def hinge_parameters() -> tuple[Scalar, Scalar]:
    """Passive hinge spring stiffness (`Scalar+[E]`) and damping (`Scalar+[E]`) pull the joints toward the home pose."""
    return mv.scalar(joint_stiffness[:, None]).field(), mv.scalar(joint_damping[:, None]).field()


def stop_parameters() -> tuple[Scalar, Scalar, Scalar]:
    """Lower stop angles (`Scalar+[E]`), upper stop angles (`Scalar+[E]`), and one-sided stop stiffness (`Scalar+[E]`) bound the hinge ranges in radians."""
    low_stops, high_stops = (mv.scalar(np.radians(bound)[:, None]).field() for bound in joint_ranges.T)
    return low_stops, high_stops, mv.scalar(np.full((len(joint_bodies), 1), limit_stiffness)).field()


def ground_parameters() -> tuple[Plane, Quadric, Scalar, float, float]:
    """Upward ground plane (`Plane+`), home-pose contact ball quadrics (`[n_balls] Plane+ <- Point-`), contact stiffness (`Scalar+[n_balls]`), damping, and friction parameterize level-ground contact."""
    _, _, _, centres, radii = home_pose()
    n_balls = len(radii)
    planes = stack((mv.x, mv.y, mv.z))                                         # [3] Plane+
    unit_balls = (planes * (planes & Point)).sum(axis=-1) / radii**2 - mv.w * (mv.w & Point)   # [n_balls] Plane+ <- Point-
    ball_shifts = ((mv.w ^ (centres - origin).dual()) / 2).exp()               # [n_balls] Motor+
    surfaces = ball_shifts >> unit_balls(ball_shifts << Point)                 # [n_balls] Plane+ <- Point-
    stiffness = mv.scalar(np.full((n_balls, 1), contact_stiffness_value)).field()   # Scalar+[n_balls]
    return mv.z, surfaces, stiffness, contact_damping, friction


def drop_state(heights: np.ndarray, tilts: np.ndarray, spins: np.ndarray) -> State:
    """Initial `State` places `n_robots` humanoids at `heights` above the ground with roll `tilts` and vertical angular `spins`."""
    n_robots = len(heights)
    lifts = (mv.zw * (heights / 2)).exp()                                      # [n_robots] Motor+
    tilts_motor = (mv.yz * (tilts / 2)).exp()                                  # [n_robots] Motor+
    angles = mv.scalar(np.zeros((n_robots, len(joint_bodies), 1))).field()     # [n_robots] Scalar+[E]
    return (lifts * tilts_motor).retyped(Motor), (mv.xy * spins).retyped(Rate), angles, angles * 0


def initial_drop_state() -> State:
    """Single-humanoid initial `State` starts raised, tilted, and spinning above level ground."""
    return drop_state(np.array([drop_height]), np.array([drop_tilt]), np.array([drop_spin]))


def flight_speeds() -> np.ndarray:
    """Reproducible free-flight hinge speeds (`[n_joints]`) in radians per second match the ordering of `joint_names`."""
    return np.random.default_rng(flight_seed).uniform(-flight_speed, flight_speed, len(joint_names))


def flight_state(n_robots: int) -> State:
    """Initial `State` places `n_robots` humanoids at `flight_height` with zero root rate and shared `flight_speeds()`."""
    lifts = (mv.zw * (np.full(n_robots, flight_height) / 2)).exp()             # [n_robots] Motor+
    angles = mv.scalar(np.zeros((n_robots, len(joint_names), 1))).field()      # [n_robots] Scalar+[E]
    speeds = mv.scalar(np.broadcast_to(flight_speeds()[:, None], (n_robots, len(joint_names), 1))).field()
    return lifts.retyped(Motor), mv.bivector(np.zeros((n_robots, 6))).retyped(Rate), angles, speeds


# --- plumbing -------------------------------------------------------------------------
def throughput(counts: np.ndarray, n_timed_frames: int) -> np.ndarray:
    """Compiled free-flight humanoid steps per second (`[n_counts]`) measure batched execution across `counts` after ahead-of-time compilation."""
    import time

    import jax

    tree, home_axes, inertias, mass_points = humanoid_model()
    gravity = -mv.z.dual() * gravity_strength                                  # [] Point-
    stiffness, damping = hinge_parameters()
    hinge_stiffness: Hinge = stiffness * Scalar.dual()                         # (Force- <- Scalar+)[E]
    hinge_damping: Hinge = damping * Scalar.dual()                             # (Force- <- Scalar+)[E]
    step_fn = partial(
        core.free_step, tree, home_axes, inertias, mass_points, gravity, hinge_stiffness, hinge_damping
    )
    step_rates = []
    for n_robots in counts:
        state = jax.device_put(flight_state(int(n_robots)), jax.devices()[0])
        compiled = core.advance.lower(step_fn, state, dt, n_steps).compile()
        state = jax.block_until_ready(compiled(state, dt))
        start_time = time.perf_counter()
        for _ in range(n_timed_frames):
            state = compiled(state, dt)
        jax.block_until_ready(state)
        step_rates.append(n_robots * n_steps * n_timed_frames / (time.perf_counter() - start_time))
    return np.array(step_rates)


def main() -> None:
    from examples.mechanics.articulated import render
    from examples.util.animation import save_animation, save_figure

    tree, home_axes, inertias, mass_points = humanoid_model()
    _, _, _, centres, radii = home_pose()
    gravity = -mv.z.dual() * gravity_strength                                  # [] Point-
    stiffness, damping = hinge_parameters()
    hinge_stiffness = stiffness * Scalar.dual()                                # (Force- <- Scalar+)[E]
    hinge_damping = damping * Scalar.dual()                                    # (Force- <- Scalar+)[E]
    low_stops, high_stops, stop_raw = stop_parameters()
    stop_stiffness = stop_raw * Scalar.dual()                                  # (Force- <- Scalar+)[E]
    ground, surfaces, contact_stiffness, ground_damping, ground_friction = ground_parameters()

    step_fn = partial(
        core.drop_step,
        tree,
        home_axes,
        inertias,
        mass_points,
        gravity,
        hinge_stiffness,
        hinge_damping,
        low_stops,
        high_stops,
        stop_stiffness,
        ground,
        surfaces,
        contact_stiffness,
        ground_damping,
        ground_friction,
    )
    forward_fn = partial(core.forward_kinematics, tree, home_axes)
    states = []

    def recorded_states():
        for current in core.trajectory(step_fn, initial_drop_state(), dt, n_steps, n_frames):
            states.append(current)
            yield current

    frames = render.animate_states(forward_fn, tree[4], centres, radii, len(capsule_bodies), recorded_states())
    save_animation(frames, "articulated", frame_duration_ms, colors=64)
    step_rates = throughput(robot_counts, n_timed)
    save_figure(render.throughput(robot_counts, step_rates, 1 / dt), "articulated_throughput")

    # checks
    motor, rate, angles, speeds = states[-1]
    motors = forward_fn(motor, angles)
    weights = gravity & (motors >> mass_points)                                # [1] Forque-[V]
    forque, forces = core.inverse_dynamics(
        tree, home_axes, inertias, motors, rate, speeds, rate, speeds, weights
    )
    loaded = weights + tree[0].adjugate() * forque[..., None].field()          # [1] Forque-[V]
    recovered_accelerations = core.forward_dynamics(
        tree, home_axes, inertias, motors, rate, speeds, forces, loaded
    )[1]
    assert ((recovered_accelerations - speeds).norm() < 1e-8).all()

    energies = stack([
        core.mechanical_energy(
            tree,
            home_axes,
            inertias,
            mass_points,
            gravity,
            hinge_stiffness,
            low_stops,
            high_stops,
            stop_stiffness,
            ground,
            surfaces,
            contact_stiffness,
            state,
        )
        for state in states
    ])                                                                         # [n_frames, 1] Scalar+
    assert (((energies[1:] - energies[:-1]) / energies[0].abs()) < 0.03).all()
    assert ((energies[-1] / energies[0]) < 0.25).all()
    assert np.isfinite(step_rates).all() and (step_rates > 0).all()


if __name__ == "__main__":
    main()
