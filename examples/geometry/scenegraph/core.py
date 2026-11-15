"""Five articulated robot-arm links and a compound two-lens camera compose into one projective map per body from canonical unit-cube vertices to sensor pixels.

Every stage of the pipeline is a linear map in 3D projective geometric algebra. Anisotropic scaling stretches points as `Point <- Point`, cumulative joint motors place each link via `pose >> scale`, joining scene points with the pupil forms incoming sight rays `Point & pupil`, two thin lenses refract rays through `Line - (center & (Line ^ plane)) / focal_length`, meeting `sensor_plane` lands rays on the sensor, and a viewport map converts metric sensor coordinates to raster pixels. Composing those maps before touching any vertex collapses kinematics, compound optics, and rasterization into one map per body.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import PGA3D

ga = PGA3D
ctx = NumpyContext(ga)
mv = ctx.multivector

Plane = ga.gatype.vector()                      # Vector+
Line = ga.gatype.antibivector()                 # Antibivector-
Point = ga.gatype.antivector()                  # Antivector-
Motor = ga.gatype.rotor()                       # Rotor+
PointMap = ga.gatype((Point, Point))            # Point- <- Point-
LineMap = ga.gatype((Line, Line))               # Line- <- Line-

origin = mv.w.dual().normalized()               # [] Point-
optical_axis = (origin & mv.z.dual()).normalized()  # [] Line-


# --- math -----------------------------------------------------------------------------
def aim_motor(eye: Point, target: Point) -> Motor:
    """Camera motor (`[] Motor+`) translating `origin` to `eye` and turning `optical_axis` onto `eye & target`."""
    unit_eye = eye.normalized()                                                # [] Point-
    translation = (unit_eye / origin).square_root()                            # [] Motor+
    sight_line = (unit_eye & target).normalized()                              # [] Line-
    shifted_axis = translation >> optical_axis                                 # [] Line-
    return (sight_line / shifted_axis).square_root() * translation             # [] Motor+


def anisotropic_scale(scale_x: float, scale_y: float, scale_z: float) -> PointMap:
    """Linear point map (`[] Point- <- Point-`) stretching `(x, y, z)` coordinates along the principal axes while preserving point weight."""
    # Pairing with the four coordinate planes reads homogeneous coordinates and sends the weight component back to the origin.
    coordinate_planes = mv.vector(np.eye(4))                                   # [4] Plane+
    scaled_axes = mv.vector(np.diag([scale_x, scale_y, scale_z, 1.0])).dual()  # [4] Point-
    return (scaled_axes * (coordinate_planes & Point)).sum(axis=0)             # [] Point- <- Point-


def robot_arm(joint_angles: tuple) -> tuple[PointMap, Motor]:
    """Body placement maps (`[5] Point- <- Point-`) and cumulative joint pivots (`[4] Motor+`) for the five-part articulated arm."""
    theta_base, theta_shoulder, theta_elbow, theta_wrist = joint_angles

    # Accumulate joint motors from the floor pedestal through turret yaw and shoulder, elbow, and wrist pitch.
    pedestal_pose = (mv.zw * 0.125 / 2).exp()                                  # [] Motor+
    pivot_turret = (mv.zw * 0.25 / 2).exp() * (mv.xy * theta_base / 2).exp()   # [] Motor+
    pivot_shoulder = pivot_turret * (mv.zw * 0.35 / 2).exp() * (mv.zx * theta_shoulder / 2).exp()  # [] Motor+
    pivot_elbow = pivot_shoulder * (mv.zw * 1.2 / 2).exp() * (mv.zx * theta_elbow / 2).exp()       # [] Motor+
    pivot_wrist = pivot_elbow * (mv.zw * 1.0 / 2).exp() * (mv.zx * theta_wrist / 2).exp()          # [] Motor+

    body_poses = stack([
        pedestal_pose,
        pivot_turret * (mv.zw * 0.175 / 2).exp(),
        pivot_shoulder * (mv.zw * 0.6 / 2).exp(),
        pivot_elbow * (mv.zw * 0.5 / 2).exp(),
        pivot_wrist * (mv.zw * 0.15 / 2).exp(),
    ])                                                                         # [5] Motor+
    body_scales = stack([
        anisotropic_scale(0.9, 0.9, 0.25),
        anisotropic_scale(0.5, 0.5, 0.35),
        anisotropic_scale(0.24, 0.24, 1.2),
        anisotropic_scale(0.18, 0.18, 1.0),
        anisotropic_scale(0.28, 0.14, 0.3),
    ])                                                                         # [5] Point- <- Point-
    bodies = body_poses >> body_scales                                         # [5] Point- <- Point-
    pivots = stack([pivot_turret, pivot_shoulder, pivot_elbow, pivot_wrist])   # [4] Motor+
    return bodies, pivots


def thin_lens(center: Point, plane: Plane, focal_length: float) -> LineMap:
    """Thin-lens refraction map (`[] Line- <- Line-`) shearing each ray proportionally to its join with `center` where it meets `plane`."""
    return Line - (center & (Line ^ plane)) / focal_length                     # [] Line- <- Line-


def lens_train(focal_front: float, focal_rear: float, rear_gap: float) -> tuple[LineMap, LineMap, Plane]:
    """Front objective lens, displaced rear relay lens (`[] Line- <- Line-`), and rear lens plane (`[] Plane+`) along the optical axis."""
    front_plane = -mv.z                                                        # [] Plane+
    front_lens = thin_lens(origin, front_plane, focal_front)                   # [] Line- <- Line-
    rear_placement = (mv.zw * rear_gap / 2).exp()                              # [] Motor+
    rear_lens = rear_placement >> thin_lens(origin, front_plane, focal_rear)(rear_placement << Line)  # [] Line- <- Line-
    return front_lens, rear_lens, rear_placement >> front_plane


def lens_camera(
    camera_pose: Motor, front_lens: LineMap, rear_lens: LineMap, pupil: Point, sensor_plane: Plane,
) -> PointMap:
    """Compound two-lens camera map (`[] Point- <- Point-`) sending world points through `pupil` and both lenses onto `sensor_plane` in camera coordinates."""
    incoming_rays = (camera_pose << Point) & pupil                             # [] Line- <- Point-
    return rear_lens(front_lens(incoming_rays)) ^ sensor_plane                 # [] Point- <- Point-


def viewport_map(width: int, height: int, sensor_width: float, sensor_height: float) -> PointMap:
    """Sensor-to-raster map (`[] Point- <- Point-`) scaling metric sensor coordinates `(x, y)` to pixel coordinates `(u, v)` with rows increasing downward."""
    sensor_scale = anisotropic_scale(width / sensor_width, -height / sensor_height, 1.0)  # [] Point- <- Point-
    centre_shift = ((mv.xw * width / 2 + mv.yw * height / 2) / 2).exp()        # [] Motor+
    return centre_shift >> sensor_scale                                        # [] Point- <- Point-


def ray_waypoints(
    scene_points: Point,
    camera_pose: Motor,
    front_lens: LineMap,
    rear_lens: LineMap,
    rear_plane: Plane,
    pupil: Point,
    sensor_plane: Plane,
) -> tuple[Point, Point, Point, Point]:
    """World-space waypoints (`[...] Point-` each) where rays from `scene_points` cross `pupil`, `rear_plane`, and `sensor_plane`."""
    incoming_rays = (camera_pose << scene_points) & pupil                      # [...] Line-
    refracted_rays = front_lens(incoming_rays)                                 # [...] Line-
    rear_hits = refracted_rays ^ rear_plane                                    # [...] Point-
    sensor_hits = rear_lens(refracted_rays) ^ sensor_plane                     # [...] Point-
    return (
        scene_points,
        (camera_pose >> pupil).broadcast_to(scene_points.shape),
        camera_pose >> (rear_hits / (mv.w & rear_hits)),
        camera_pose >> (sensor_hits / (mv.w & sensor_hits)),
    )


# --- plumbing -------------------------------------------------------------------------
def canonical_unit_box() -> Point:
    """Eight corner points (`[8] Point-`) of the canonical unit cube centred on the origin."""
    return mv.point([
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ])                                                                         # [8] Point-


def unit_box_topology() -> tuple[np.ndarray, np.ndarray]:
    """Twelve edge index pairs (`[12, 2]`) and six quad face index rows (`[6, 4]`) of the canonical unit cube."""
    edges = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ], dtype=np.int32)

    faces = np.array([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [0, 3, 7, 4],
        [1, 2, 6, 5],
    ], dtype=np.int32)

    return edges, faces
