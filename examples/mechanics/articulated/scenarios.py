"""Scenes for the articulated robot: Brax's humanoid, its bodies and hinges, dropped limp onto level ground, where it
lands and slumps into a heap; and flying free, as its compiled step is timed for batches of humanoids."""

from __future__ import annotations

from collections.abc import Iterator
from functools import partial

import numpy as np

from numga import stack
from numga.sparse import SparseExtensor
from examples.mechanics.articulated import core
from examples.mechanics.articulated.core import Motor, Point, Rate, Scalar, mv
from examples.util.topology import Topology

# Brax's humanoid, its bodies first: each by its parent, before it, and its origin in its parent's frame, in metres. The
# torso is the root; then the lower waist and the pelvis, each leg's thigh, shin and foot, and each arm's upper and
# lower arm.
body_parents = np.array([-1, 0, 1, 2, 3, 4, 2, 6, 7, 0, 9, 0, 11])
body_offsets = np.array([[0, 0, 0], [-.01, 0, -.26], [0, 0, -.165], [0, -.1, -.04], [0, .01, -.403], [0, 0, -.45],
                         [0, .1, -.04], [0, -.01, -.403], [0, 0, -.45], [0, -.17, .06], [.18, -.18, -.18],
                         [0, .17, .06], [.18, .18, -.18]])
# Its hinges, in its order, each by the body it turns, its pivot in that body's frame and its axis's direction there;
# its stiffness, in newton metres per radian, and damping, in newton metre seconds per radian; and its range, in
# degrees: the abdomen's three, each hip's three and knee, each shoulder's two and elbow.
joint_bodies = np.array([1, 1, 2, 3, 3, 3, 4, 6, 6, 6, 7, 9, 9, 10, 11, 11, 12])
joint_pivots = np.array([[0, 0, .065], [0, 0, .065], [0, 0, .1]] + [[0, 0, 0]] * 3 + [[0, 0, .02]] + [[0, 0, 0]] * 3
                        + [[0, 0, .02]] + [[0, 0, 0]] * 6)
joint_directions = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, -1, 0],
                             [-1, 0, 0], [0, 0, -1], [0, 1, 0], [0, -1, 0], [2, 1, 1], [0, -1, 1], [0, -1, 1],
                             [2, -1, 1], [0, 1, 1], [0, -1, -1]])
joint_stiffness = np.array([20, 10, 10, 10, 10, 20, 0, 10, 10, 20, 1, 1, 1, 0, 1, 1, 0])
joint_damping = np.array([5, 5, 5, 5, 5, 5, 1, 5, 5, 5, 1, 1, 1, 1, 1, 1, 1])
joint_ranges = np.array([[-45, 45], [-75, 30], [-35, 35], [-25, 5], [-60, 35], [-110, 20], [-160, -2], [-25, 5],
                         [-60, 35], [-110, 20], [-160, -2], [-85, 60], [-85, 60], [-90, 50], [-60, 85], [-60, 85],
                         [-90, 50]])
# Its capsules, each by the body it belongs to, the ends of its axis in that body's frame and its radius, in metres:
# the torso and the upper waist, the lower waist, the buttocks, each leg's thigh and shin, and each arm's upper and
# lower arm.
capsule_bodies = np.array([0, 0, 1, 2, 3, 4, 6, 7, 9, 10, 11, 12])
capsule_starts = np.array([[0, -.07, 0], [-.01, -.06, -.12], [0, -.06, 0], [-.02, -.07, 0], [0, 0, 0], [0, 0, 0],
                           [0, 0, 0], [0, 0, 0], [0, 0, 0], [.01, .01, .01], [0, 0, 0], [.01, -.01, .01]])
capsule_ends = np.array([[0, .07, 0], [-.01, .06, -.12], [0, .06, 0], [-.02, .07, 0], [0, .01, -.34], [0, 0, -.3],
                         [0, -.01, -.34], [0, 0, -.3], [.16, -.16, -.16], [.17, .17, .17], [.16, .16, -.16],
                         [.17, -.17, .17]])
capsule_radii = np.array([.07, .06, .06, .09, .06, .049, .06, .049, .04, .031, .04, .031])
# Its spheres, each by its body, its centre in that body's frame and its radius: the head, the feet and the hands. Each
# part, capsule or sphere, is solid, at MuJoCo's default density, in kilograms per cubic metre.
sphere_bodies = np.array([0, 5, 8, 10, 12])
sphere_centres = np.array([[0, 0, .19], [0, 0, .1], [0, 0, .1], [.18, .18, .18], [.18, -.18, .18]])
sphere_radii = np.array([.09, .075, .075, .04, .04])
density = 1000.0
# The stiffness of a joint's stops, in newton metres per radian, as Brax's limits; gravity, in metres per second squared.
limit_stiffness = 2500.0
gravity_strength = 9.81
# The ground's push on a part per depth below it, in newtons per metre, the share it grows by per speed of sinking, in
# seconds per metre, and its friction per push and speed of slip, in seconds per metre.
contact_stiffness, contact_damping, friction = 3e4, 3.0, 30.0
# The torso's height above the ground when dropped, in metres, its tilt about x, in radians, and its spin about the
# vertical, in radians per second.
drop_height, drop_tilt, drop_spin = 1.45, 0.15, 1.0
# The time step of the linearly implicit Euler steps, in seconds, the steps a frame, the frames, and the duration of a
# frame in milliseconds: a frame's steps span its duration.
dt = 0.01
n_steps = 4
n_frames = 60
frame_duration = 40
# The humanoid flying free, as its step is timed: its hinges by name in Brax's order, its torso as high as Brax's,
# in metres, at rest, and its hinges at angle zero, turning at a fixed random set of speeds up to a largest, in radians
# per second, drawn from a seed.
joint_names = ["abdomen_z", "abdomen_y", "abdomen_x", "right_hip_x", "right_hip_z", "right_hip_y", "right_knee",
               "left_hip_x", "left_hip_z", "left_hip_y", "left_knee", "right_shoulder1", "right_shoulder2",
               "right_elbow", "left_shoulder1", "left_shoulder2", "left_elbow"]
flight_height, flight_speed, flight_seed = 1.4, 0.5, 7
# The batches timed, from one humanoid to a few thousand, and the frames each is timed over after one to compile it.
robot_counts = 4 ** np.arange(7)
n_timed = 5
# The intrinsic scalar one: a join of points, intrinsic, times it is the line through them, extrinsic.
flip = mv.antipseudoscalar([1.0])                                              # [] Antipseudoscalar-
origin = mv.w.dual()                                                           # [] Point-


# --- math -----------------------------------------------------------------------------
def humanoid() -> tuple[Topology, np.ndarray, np.ndarray, int]:
    """The humanoid's tree, an edge for each hinge from the link before it to the link it turns, with a massless link
    between each two hinges of one body; the link that carries each body, `[bodies]`, a foot its shin's; each body's
    origin in the home pose, `[bodies, 3]`; and how many hinges the longest path from the torso passes."""
    origins, carriers, depths, edges = np.zeros((len(body_parents), 3)), np.zeros(len(body_parents), dtype=int), [0], []
    for body, parent in enumerate(body_parents[1:], start=1):
        origins[body] = origins[parent] + body_offsets[body]
        link = carriers[parent]
        for _ in range(np.count_nonzero(joint_bodies == body)):
            edges.append([link, len(depths)])
            depths.append(depths[link] + 1)
            link = len(depths) - 1
        carriers[body] = link
    return Topology(len(depths), np.array(edges)), carriers, origins, max(depths)


def principal_samples(reaches: np.ndarray) -> np.ndarray:
    """Six points about the origin, `[..., 6, 3]`, at plus and minus each reach `[..., 3]` along x, y and z: a sixth of
    a mass at each has the mass's second moment along each axis, a third of the mass times the reach's square."""
    offsets = reaches[..., None, :] * np.eye(3)                               # [..., 3, 3]
    return np.concatenate([offsets, -offsets], axis=-2)


def home_pose() -> tuple[Rate, Point, np.ndarray, Point, np.ndarray]:
    """The humanoid standing in its home pose, its torso at the origin and every hinge at angle zero: each hinge's axis,
    `Rate[joints]`; six mass samples of each part, the capsules then the spheres, `[parts, 6] Point`, with the part's
    mass and second moments, and their masses `[parts, 1]`; and the balls a flat ground can meet, `[balls] Point`, the
    capsules' starts, then their ends, then the spheres, and their radii `[balls]`."""
    _, _, origins, _ = humanoid()
    planes = stack((mv.x, mv.y, mv.z))                                         # [3] Plane+
    # Each hinge's axis, the turn about the line through its pivot along its direction, by the right hand: the direction
    # joined with the pivot, made extrinsic by the intrinsic scalar one.
    pivots = mv.point(origins[joint_bodies] + joint_pivots)                   # [joints] Point-
    directions = mv.direction(joint_directions / np.linalg.norm(joint_directions, axis=-1, keepdims=True))
    axes = ((directions & pivots) * flip).field()                              # Rate+[joints]
    # Each part as a capsule, a sphere one of no length: its axis's ends, its radius and its length.
    bodies = np.concatenate([capsule_bodies, sphere_bodies])
    starts = origins[bodies] + np.concatenate([capsule_starts, sphere_centres])
    ends = origins[bodies] + np.concatenate([capsule_ends, sphere_centres])
    radii = np.concatenate([capsule_radii, sphere_radii])
    lengths = np.linalg.norm(ends - starts, axis=-1)
    # Its mass, a cylinder's and two half balls', and its moments of inertia about its axis and across it.
    cylinders, balls = density * np.pi * radii**2 * lengths, density * 4 / 3 * np.pi * radii**3
    masses = cylinders + balls
    axial = cylinders * radii**2 / 2 + balls * 2 * radii**2 / 5
    transverse = cylinders * (radii**2 / 4 + lengths**2 / 12) + balls * (2 * radii**2 / 5 + lengths**2 / 4
                                                                      + 3 * lengths * radii / 8)
    # Its second moments, across its axis and along it, and how far six samples must reach to share them.
    reaches = np.sqrt(3 * np.stack([axial / 2, axial / 2, transverse - axial / 2], axis=-1) / masses[:, None])
    # The plane through the origin square to each capsule's axis, facing up, a sphere's z; the turn carrying z onto
    # it, and the shift onto the part's middle.
    upward = np.where((ends - starts)[:, 2:] < 0, starts - ends, ends - starts)
    normals = np.where(lengths[:, None] > 0, upward, [0, 0, 1])
    facing = (planes * normals).sum(axis=-1).normalized()                      # [parts] Plane+
    turns = (1 + facing * mv.z).normalized()                                   # [parts] Motor+
    shifts = ((mv.w ^ (mv.point((starts + ends) / 2) - origin).dual()) / 2).exp()   # [parts] Motor+
    samples = (shifts * turns)[:, None] >> mv.point(principal_samples(reaches))     # [parts, 6] Point-
    # The balls a flat ground can meet: each capsule's two ends, and each sphere.
    capsules = len(capsule_bodies)
    centres = mv.point(np.concatenate([starts[:capsules], ends[:capsules], starts[capsules:]]))   # [balls] Point-
    return axes, samples, masses[:, None] / 6, centres, np.concatenate([radii[:capsules], radii])


def humanoid_tree() -> tuple[core.Tree, SparseExtensor]:
    """The humanoid's tree, the balls the ground can meet carried by their links, and the link that carries each part,
    `numbers[parts <- links]`."""
    topology, carriers, _, n_generations = humanoid()
    tree = core.tree(topology, carriers[np.concatenate([capsule_bodies, capsule_bodies, sphere_bodies])], n_generations)
    return tree, topology.reading(carriers[np.concatenate([capsule_bodies, sphere_bodies])], 0)


def robot() -> tuple[core.Tree, core.Robot]:
    """The humanoid's tree, and its data in the home pose."""
    tree, parts = humanoid_tree()
    axes, samples, masses, _, _ = home_pose()
    # Each part's inertia, the map from its rate to its momentum: every sample joined with its velocity, weighted; a
    # link's, the sum of its parts', none for a massless link between two hinges.
    part_inertias = samples.regressive(samples.commutator(Rate) * masses).sum(axis=-1)   # [parts] Forque- <- Rate+
    inertias = ~parts * part_inertias.field()                                  # (Forque- <- Rate+)[links]
    mass_points = ~parts * (samples * masses).sum(axis=-1).field()             # Point-[links]
    return tree, core.Robot(axes, inertias, mass_points)


def hinges() -> tuple[Scalar, Scalar]:
    """The hinges' springs and dampers towards the home pose: their stiffness and damping, `Scalar[joints]` each."""
    return mv.scalar(joint_stiffness[:, None]).field(), mv.scalar(joint_damping[:, None]).field()


def physics() -> core.Physics:
    """Gravity downward, and the hinges' springs and dampers."""
    return core.Physics(-mv.z.dual() * gravity_strength, *hinges())


def free_loads(tree: core.Tree, robot_data: core.Robot) -> core.Loads:
    """What acts on the humanoid flying free: its links' weights and inertias, and its hinges."""
    return partial(core.own_loads, tree, robot_data, physics())


def drop_loads(tree: core.Tree, robot_data: core.Robot) -> core.Loads:
    """What acts on the humanoid dropped onto the ground: as in free flight, with the ground and the stops."""
    return partial(core.drop_loads, tree, robot_data, physics(), ground(), stops())


def ground() -> core.Ground:
    """The ground through the origin facing up, the surfaces of the balls that can meet it, and its push and
    friction."""
    _, _, _, centres, radii = home_pose()
    # Each ball's surface: the round form about the origin at its radius, shifted onto its centre.
    planes = stack((mv.x, mv.y, mv.z))                                         # [3] Plane+
    round_forms = (planes * (planes & Point)).sum(axis=-1) / radii**2 - mv.w * (mv.w & Point)   # [balls] Plane+ <- Point-
    shifts = ((mv.w ^ (centres - origin).dual()) / 2).exp()                    # [balls] Motor+
    pushes = mv.scalar(np.full((len(radii), 1), contact_stiffness)).field()    # Scalar+[balls]
    return core.Ground(mv.z, shifts >> round_forms(shifts << Point), pushes, contact_damping, friction)


def stops() -> core.Stops:
    """The stops at the ends of the hinges' ranges: the ends, in radians, and the stops' stiffness, `Scalar[joints]`
    each."""
    low, high = (mv.scalar(np.radians(bound)[:, None]).field() for bound in joint_ranges.T)   # Scalar+[joints]
    return core.Stops(low, high, mv.scalar(np.full((len(joint_bodies), 1), limit_stiffness)).field())


def drop(heights: np.ndarray, tilts: np.ndarray, spins: np.ndarray) -> core.State:
    """Humanoids `[robots]` in their home pose, their torsos raised to their heights above the ground, tilted about x
    and spinning about the vertical."""
    lift = (mv.zw * (heights / 2)).exp()                                       # [robots] Motor+
    tilt = (mv.yz * (tilts / 2)).exp()                                         # [robots] Motor+
    angles = mv.scalar(np.zeros((len(heights), len(joint_bodies), 1))).field()   # [robots] Scalar+[joints]
    # The motor and the rate of the full types the steps carry, so that one compiled step serves every frame.
    return core.State((lift * tilt).retyped(Motor), (mv.xy * spins).retyped(Rate), angles, angles * 0)


def flight_speeds() -> np.ndarray:
    """The hinges' speeds in free flight, `[joints]`, in radians per second, in the order of `joint_names`."""
    return np.random.default_rng(flight_seed).uniform(-flight_speed, flight_speed, len(joint_names))


def flight(n_robots: int) -> core.State:
    """Humanoids `[robots]` flying free from Brax's initial pose: the torso raised to Brax's height, at rest, and every
    hinge at angle zero, turning at its speed in free flight, the same for every humanoid."""
    lift = (mv.zw * (np.full(n_robots, flight_height) / 2)).exp()             # [robots] Motor+
    angles = mv.scalar(np.zeros((n_robots, len(joint_names), 1))).field()      # [robots] Scalar+[joints]
    speeds = mv.scalar(np.broadcast_to(flight_speeds()[:, None], (n_robots, len(joint_names), 1))).field()
    # The motor and the rate of the full types the steps carry, so that one compiled step serves every frame.
    return core.State(lift.retyped(Motor), mv.bivector(np.zeros((n_robots, 6))).retyped(Rate), angles, speeds)


def dropped() -> core.State:
    """The humanoid as the scene drops it: its torso raised, tilted and spinning, its hinges at rest in the home pose."""
    return drop(np.array([drop_height]), np.array([drop_tilt]), np.array([drop_spin]))


def fall(state: core.State) -> Iterator[Point]:
    """The humanoids dropped onto the ground from a state: the centre of every ball at each frame, `[robots, balls]`,
    the capsules' starts, then their ends, then the spheres, from the drop on."""
    tree, robot_data = robot()
    loads = drop_loads(tree, robot_data)
    _, _, _, centres, _ = home_pose()
    for _ in range(n_frames):
        carried = (tree.balls * core.forward_kinematics(tree, robot_data, state.motor, state.angles)).batch()
        yield carried >> centres
        state = core.advance(tree, robot_data, state, dt, n_steps, loads)


# --- plumbing -------------------------------------------------------------------------
def throughput(counts: np.ndarray, n_timed: int) -> np.ndarray:
    """How many humanoid steps a second the compiled step takes, `[counts]`, for a batch of each count of humanoids
    flying free, with their weights and their hinges alone: each batch's frame of steps compiled ahead of
    time, then timed over n_timed frames, its state committed to the device as the step's results are, so that the
    compiled step takes them as they come and would refuse a state of another type rather than compile again."""
    import time

    import jax

    tree, robot_data = robot()
    loads = free_loads(tree, robot_data)
    rates = []
    for n_robots in counts:
        state = jax.device_put(flight(n_robots), jax.devices()[0])
        compiled = core.advance.lower(tree, robot_data, state, dt, n_steps, loads).compile()
        state = jax.block_until_ready(compiled(robot_data, state, dt))
        start = time.perf_counter()
        for _ in range(n_timed):
            state = compiled(robot_data, state, dt)
        jax.block_until_ready(state)
        rates.append(n_robots * n_steps * n_timed / (time.perf_counter() - start))
    return np.array(rates)


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.mechanics.articulated import render

    _, _, _, _, radii = home_pose()
    centres = fall(dropped())
    save_animation(render.animate(centres, radii, len(capsule_bodies)), "articulated", frame_duration, colors=64)
    save_figure(render.throughput(robot_counts, throughput(robot_counts, n_timed), 1 / dt), "articulated_throughput")


if __name__ == "__main__":
    main()
