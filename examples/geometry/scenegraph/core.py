r"""A five-part articulated robot arm and a compound two-lens camera compose into one map per body from canonical unit-box vertices to sensor pixels.

Every stage of the pipeline is a linear map in projective geometric algebra. A body's stretch is a map on points, `Point <- Point`; a joint motor places it, `pose >> scale`; the camera joins each point with the pupil into a ray, `Point & pupil`, refracts the ray through two thin lenses, `Line - (center & (Line ^ plane)) / focal_length`, and meets it with the sensor, `Line ^ sensor_plane`; and a viewport scales the sensor onto pixel coordinates. Calling one map on another composes them before any vertex is touched, collapsing kinematics, optics, and rasterization into one map per body.

In the matrix notation of computer graphics the collapsed map reads as the product of the viewport, projection, view, and model matrices, $V P C M_i$, one per part, and in ray-transfer optics a thin lens as the matrix $\begin{pmatrix} 1 & 0 \\ -1/f & 1 \end{pmatrix}$ on a ray's height and angle.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import PGA3D

ga = PGA3D
ctx = NumpyContext(ga)
mv = ctx.multivector

# Planes, lines and points are the vectors, antibivectors and antivectors; motors rotate and translate.
Plane = ga.gatype.vector()                      # Vector+
Line = ga.gatype.antibivector()                 # Antibivector-
Point = ga.gatype.antivector()                  # Antivector-
Motor = ga.gatype.rotor()                       # Rotor+
PointMap = ga.gatype((Point, Point))
LineMap = ga.gatype((Line, Line))

origin = mv.w.dual().normalized()               # [] Point-
optical_axis = (origin & mv.z.dual()).normalized()  # [] Line-


# --- math -----------------------------------------------------------------------------
def look_at(eye: Point, target: Point) -> Motor:
    """The camera motor places the origin at `eye` and turns the optical axis onto `eye & target`."""
    eye = eye.normalized()                                                     # [] Point-
    translator = (eye / origin).square_root()                                  # [] Motor+
    sight = (eye & target).normalized()                                        # [] Line-
    return (sight / (translator >> optical_axis)).square_root() * translator   # [] Motor+


def anisotropic_scale(scale_x: float, scale_y: float, scale_z: float) -> PointMap:
    """The anisotropic scaling map stretches points along the principal axes while preserving their weights."""
    # The coordinate planes read a point's coordinates, mv.x & point being its x; each
    # coordinate is sent to its scaled axis direction, and the weight to the origin.
    coordinate_planes = mv.vector(np.eye(4))                                   # [coordinates] Plane+
    scaled_axes = mv.antivector(np.diag([scale_x, scale_y, scale_z, 1.0]))     # [coordinates] Point-
    return (scaled_axes * (coordinate_planes & Point)).sum(axis=0)             # [] Point- <- Point-


def robot_arm(joint_angles: tuple) -> PointMap:
    """Each body of the five-part arm is the canonical unit box scaled to shape and placed into the world by its cumulative joint motor."""
    theta_base, theta_shoulder, theta_elbow, theta_wrist = joint_angles

    # Body 0: Base pedestal resting on the floor z == 0
    body_0 = (mv.zw * 0.125 / 2).exp() >> anisotropic_scale(0.9, 0.9, 0.25)    # [] Point- <- Point-

    # Turret base pivot at height 0.25, yawing around the vertical z-axis:
    pivot_turret = (mv.zw * 0.25 / 2).exp() * (mv.xy * theta_base / 2).exp()   # [] Motor+
    body_1 = (pivot_turret * (mv.zw * 0.175 / 2).exp()) >> anisotropic_scale(0.5, 0.5, 0.35)

    # Shoulder pivot 0.35 above turret base, pitching around the local y-axis:
    pivot_shoulder = pivot_turret * (mv.zw * 0.35 / 2).exp() * (mv.zx * theta_shoulder / 2).exp()
    body_2 = (pivot_shoulder * (mv.zw * 0.6 / 2).exp()) >> anisotropic_scale(0.24, 0.24, 1.2)

    # Elbow pivot at the tip of the upper arm (1.2 units along it), pitching around the local y-axis:
    pivot_elbow = pivot_shoulder * (mv.zw * 1.2 / 2).exp() * (mv.zx * theta_elbow / 2).exp()
    body_3 = (pivot_elbow * (mv.zw * 0.5 / 2).exp()) >> anisotropic_scale(0.18, 0.18, 1.0)

    # Wrist pivot at the tip of the forearm (1.0 unit along it), pitching around the local y-axis:
    pivot_wrist = pivot_elbow * (mv.zw * 1.0 / 2).exp() * (mv.zx * theta_wrist / 2).exp()
    body_4 = (pivot_wrist * (mv.zw * 0.15 / 2).exp()) >> anisotropic_scale(0.28, 0.14, 0.3)

    return stack([body_0, body_1, body_2, body_3, body_4])                     # [bodies] Point- <- Point-


def thin_lens(center: Point, plane: Plane, focal_length: float) -> LineMap:
    """A thin lens shears each ray proportionally to its incidence with the optical centre."""
    return Line - (center & (Line ^ plane)) / focal_length                     # [] Line- <- Line-


def lens_train(focal_front: float, focal_rear: float, rear_gap: float) -> tuple[LineMap, LineMap, Plane]:
    """The front and rear thin lenses sit on the optical axis -z, with the rear lens carried along the axis by its displacement motor."""
    front_plane = -mv.z                                                        # [] Plane+
    front_lens = thin_lens(origin, front_plane, focal_front)                   # [] Line- <- Line-
    rear_placement = (mv.zw * rear_gap / 2).exp()                              # [] Motor+
    rear_lens = rear_placement >> thin_lens(origin, front_plane, focal_rear)(rear_placement << Line)  # [] Line- <- Line-
    return front_lens, rear_lens, rear_placement >> front_plane


def lens_camera(camera_pose: Motor, front_lens: LineMap, rear_lens: LineMap, pupil: Point, sensor_plane: Plane) -> PointMap:
    """The compound two-lens camera maps world points through the pupil and both lenses onto the sensor plane in the camera frame."""
    incoming_rays = (camera_pose << Point) & pupil                             # [] Line- <- Point-
    return rear_lens(front_lens(incoming_rays)) ^ sensor_plane                 # [] Point- <- Point-


def viewport(width: int, height: int, sensor_width: float, sensor_height: float) -> PointMap:
    """The viewport map scales and shifts metric sensor coordinates (x, y) to raster pixel coordinates (u, v) with rows increasing downward."""
    scale = anisotropic_scale(width / sensor_width, -height / sensor_height, 1.0)
    return ((mv.xw * width / 2 + mv.yw * height / 2) / 2).exp() >> scale       # [] Point- <- Point-


def trace_rays(
    scene_points: Point,
    camera_pose: Motor,
    front_lens: LineMap,
    rear_lens: LineMap,
    rear_plane: Plane,
    pupil: Point,
    sensor_plane: Plane,
) -> tuple[Point, Point, Point, Point]:
    """Each ray from a scene point through the pupil crosses the front lens, the rear lens plane, and the sensor plane at these world points."""
    ray_in = (camera_pose << scene_points) & pupil                             # [rays] Line-
    ray_mid = front_lens(ray_in)                                               # [rays] Line-
    rear_hit = ray_mid ^ rear_plane                                            # [rays] Point-
    sensor_hit = rear_lens(ray_mid) ^ sensor_plane                             # [rays] Point-
    return (
        scene_points,
        (camera_pose >> pupil).broadcast_to(scene_points.shape),
        camera_pose >> (rear_hit / (mv.w & rear_hit)),
        camera_pose >> (sensor_hit / (mv.w & sensor_hit)),
    )


# --- plumbing -------------------------------------------------------------------------
def canonical_unit_box() -> Point:
    """The eight vertices of the canonical unit cube are centred on the origin."""
    return mv.point([
        [-0.5, -0.5, -0.5],
        [ 0.5, -0.5, -0.5],
        [ 0.5,  0.5, -0.5],
        [-0.5,  0.5, -0.5],
        [-0.5, -0.5,  0.5],
        [ 0.5, -0.5,  0.5],
        [ 0.5,  0.5,  0.5],
        [-0.5,  0.5,  0.5],
    ])


def unit_box_topology() -> tuple[np.ndarray, np.ndarray]:
    """The canonical unit cube has twelve edges and six quad faces."""
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
