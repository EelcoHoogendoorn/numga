"""PGA3D scenegraph, robot kinematics, and compound multi-lens camera optics.

All mathematical operations are formulated as coordinate-free extensors (linear maps
between blade subspaces). The entire visual pipeline of articulated forward kinematics,
anisotropic box scaling, camera pose transformation, pupil ray formation, compound
two-lens refraction, sensor plane intersection, and viewport rasterization collapses
into a single compiled extensor mapping canonical unit box vertices to 2D screen pixels.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import PGA3D

# --- algebra context and types --------------------------------------------------------
ga = PGA3D
ctx = NumpyContext(ga)
mv = ctx.multivector

# Planes, lines and points are the vectors, bivectors and antivectors; motors rotate and translate.
Scalar = ga.gatype.scalar()
Plane = ga.gatype.vector()
Line = ga.gatype.bivector()
Point = ga.gatype.antivector()
Motor = ga.gatype.rotor()

# Extensors (linear maps between blade subspaces): collineations of points, and optical maps of rays.
PointMap = ga.gatype((Point, Point))        # Point <- Point
LineMap = ga.gatype((Line, Line))           # Line <- Line

origin = mv.zyx


def point(coords: np.ndarray) -> Point:
    """Finite points at (..., 3) coordinates: the dual of the homogeneous vector."""
    return (mv("x y z", coords) + mv.w).dual()


# --- geometry primitives --------------------------------------------------------------
def canonical_unit_box() -> Point:
    """The 8 vertices of the canonical unit box: a unit cube centred on the origin."""
    return point(np.array([
        [-0.5, -0.5, -0.5],
        [ 0.5, -0.5, -0.5],
        [ 0.5,  0.5, -0.5],
        [-0.5,  0.5, -0.5],
        [-0.5, -0.5,  0.5],
        [ 0.5, -0.5,  0.5],
        [ 0.5,  0.5,  0.5],
        [-0.5,  0.5,  0.5],
    ]))


def unit_box_topology() -> tuple[np.ndarray, np.ndarray]:
    """Return (edges, faces) vertex index arrays for the unit box."""
    edges = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom ring (z == -0.5)
        [4, 5], [5, 6], [6, 7], [7, 4],  # top ring (z == 0.5)
        [0, 4], [1, 5], [2, 6], [3, 7],  # vertical pillars
    ], dtype=np.int32)

    faces = np.array([
        [0, 1, 2, 3],  # bottom (-z)
        [4, 5, 6, 7],  # top (+z)
        [0, 1, 5, 4],  # front (-y)
        [2, 3, 7, 6],  # back (+y)
        [0, 3, 7, 4],  # left (-x)
        [1, 2, 6, 5],  # right (+x)
    ], dtype=np.int32)

    return edges, faces


# --- affine and kinematic maps --------------------------------------------------------
def anisotropic_scale(scale_x: float, scale_y: float, scale_z: float) -> PointMap:
    """Anisotropic scaling extensor (Point <- Point) along the principal axes.

    The coordinate planes read a point's coordinates, `mv.x & point` being its x; each
    coordinate is sent to its scaled axis direction, and the weight to the origin.
    """
    coordinate_planes = mv("x y z w", np.eye(4))                              # [coordinates] Plane
    scaled_axes = mv("yzw zxw xyw zyx", np.diag([scale_x, scale_y, scale_z, 1.0]))   # [coordinates] Point
    return (scaled_axes * (coordinate_planes & Point)).sum(axis=0)


# --- articulated robot arm forward kinematics -----------------------------------------
def robot_arm(joint_angles: tuple) -> tuple[PointMap, list[Motor]]:
    """Forward kinematics of a 5-body articulated robot arm.

    Every body is the canonical unit box shaped by an anisotropic scaling extensor and
    placed into the world by its cumulative joint motor. The joint angles are base yaw,
    shoulder pitch, elbow pitch and wrist pitch; the result is the batch of five
    unit-box-to-world maps and the four joint pivots.
    """
    theta_base, theta_shoulder, theta_elbow, theta_wrist = joint_angles
    # A translator by d along z is (mv.zw * d / 2).exp(), a rotor by theta about z is (mv.xy * theta / 2).exp().

    # Body 0: Base pedestal (fixed stationary block resting on the floor z == 0)
    body_0 = (mv.zw * 0.125 / 2).exp() >> anisotropic_scale(0.9, 0.9, 0.25)

    # Turret base pivot at height 0.25, yawing around vertical z-axis:
    pivot_turret = (mv.zw * 0.25 / 2).exp() * (mv.xy * theta_base / 2).exp()

    # Body 1: Turret rotating body
    body_1 = (pivot_turret * (mv.zw * 0.175 / 2).exp()) >> anisotropic_scale(0.5, 0.5, 0.35)

    # Shoulder pivot 0.35 above turret base, pitching around local y-axis:
    pivot_shoulder = pivot_turret * (mv.zw * 0.35 / 2).exp() * (mv.zx * theta_shoulder / 2).exp()

    # Body 2: Upper arm (anisotropically elongated along arm length)
    body_2 = (pivot_shoulder * (mv.zw * 0.6 / 2).exp()) >> anisotropic_scale(0.24, 0.24, 1.2)

    # Elbow pivot at tip of upper arm (1.2 units along upper arm), pitching around local y-axis:
    pivot_elbow = pivot_shoulder * (mv.zw * 1.2 / 2).exp() * (mv.zx * theta_elbow / 2).exp()

    # Body 3: Forearm (anisotropically elongated along forearm length)
    body_3 = (pivot_elbow * (mv.zw * 0.5 / 2).exp()) >> anisotropic_scale(0.18, 0.18, 1.0)

    # Wrist pivot at tip of forearm (1.0 units along forearm), pitching around local y-axis:
    pivot_wrist = pivot_elbow * (mv.zw * 1.0 / 2).exp() * (mv.zx * theta_wrist / 2).exp()

    # Body 4: Gripper tool block / end-effector
    body_4 = (pivot_wrist * (mv.zw * 0.15 / 2).exp()) >> anisotropic_scale(0.28, 0.14, 0.3)

    # Stack all 5 body extensors into a single batched extensor:
    bodies_to_world = stack([body_0, body_1, body_2, body_3, body_4])
    joint_pivots = [pivot_turret, pivot_shoulder, pivot_elbow, pivot_wrist]
    return bodies_to_world, joint_pivots


# --- compound multi-lens camera optics ------------------------------------------------
def thin_lens(center: Point, plane: Plane, focal_length: float) -> LineMap:
    """Thin lens extensor (Line <- Line) focusing rays towards its center.

    Following Gaussian optics in PGA, a thin lens shears a line proportionally to its
    incidence with the optical center: `Line - (center & (Line ^ plane)) / focal_length`.
    """
    return Line - (center & (Line ^ plane)) / focal_length


def lens_train(focal_front: float, focal_rear: float, rear_gap: float):
    """Front and rear lenses on the optical axis -z, and the rear lens plane.

    The front lens sits at the origin. The rear lens is placed along the optical axis by
    transforming the lens extensor with its displacement motor:
    `rear_lens = rear_placement >> thin_lens(origin, front_plane, focal_rear)(rear_placement << Line)`.
    """
    front_plane = -mv.z
    front_lens = thin_lens(origin, front_plane, focal_front)
    rear_placement = (mv.zw * rear_gap / 2).exp()
    rear_lens = rear_placement >> thin_lens(origin, front_plane, focal_rear)(rear_placement << Line)
    return front_lens, rear_lens, rear_placement >> front_plane


def lens_camera(camera_pose: Motor, front_lens: LineMap, rear_lens: LineMap, pupil: Point, sensor_plane: Plane) -> PointMap:
    """Compound multi-lens camera extensor mapping world points to sensor points in the camera frame.

    A compound optical system is the functional composition of the lens extensors,
    optics = rear_lens(front_lens).
    """
    # Transform world points into local camera space, then join with pupil to form rays:
    incoming_rays = (camera_pose << Point) & pupil     # Line <- Point
    to_sensor = Line ^ sensor_plane                    # Point <- Line
    return to_sensor(rear_lens(front_lens(incoming_rays)))


def look_at(position: np.ndarray, target: np.ndarray) -> Motor:
    """Camera motor in world space whose optical axis -z points towards the target."""
    translation = (mv("xw yw zw", position) / 2).exp()
    # Pitch the optical axis from horizontal towards the target: np.pi / 2 about x turns -z
    # forward (+y), minus the pitch tilts it downward.
    delta = target - position
    pitch = np.arctan2(delta[2], delta[1])
    return translation * (mv.yz * (np.pi / 2 - pitch) / 2).exp()


# --- viewport and scenegraph collapse -------------------------------------------------
def viewport(width: int, height: int, sensor_width: float, sensor_height: float) -> PointMap:
    """Viewport extensor mapping metric sensor coordinates (x, y) to pixel coordinates (u, v).

    The physical sensor, sensor_width by sensor_height, is mapped to width by height pixels.
    Because raster pixel rows increase downward (row 0 at the top), the vertical axis is reflected.
    """
    scale = anisotropic_scale(width / sensor_width, -height / sensor_height, 1.0)
    return ((mv.xw * width / 2 + mv.yw * height / 2) / 2).exp() >> scale


def project_vertices(local_to_pixel: PointMap, unit_box: Point) -> Point:
    """Batch-project canonical unit box vertices to screen pixels with unit weight."""
    raw_pixels = local_to_pixel[:, None](unit_box[None, :])
    # Perspective normalization:
    return raw_pixels / (mv.w & raw_pixels)


# --- 3D optical ray tracing -----------------------------------------------------------
def trace_rays(
    scene_points: Point,
    camera_pose: Motor,
    front_lens: LineMap,
    rear_lens: LineMap,
    rear_plane: Plane,
    pupil: Point,
    sensor_plane: Plane,
):
    """The world points where rays from scene points through the pupil cross each optical interface.

    Returns, per scene point, the scene point, the pupil on the front lens, the hit on the
    rear lens plane, and the hit on the sensor.
    """
    ray_in = (camera_pose << scene_points) & pupil

    # Refraction through front lens:
    ray_mid = front_lens(ray_in)
    rear_hit = ray_mid ^ rear_plane

    # Refraction through rear lens:
    sensor_hit = rear_lens(ray_mid) ^ sensor_plane

    # Transform all points back to world space:
    return (
        scene_points,
        (camera_pose >> pupil).broadcast_to(scene_points.shape),
        camera_pose >> (rear_hit / (mv.w & rear_hit)),
        camera_pose >> (sensor_hit / (mv.w & sensor_hit)),
    )
