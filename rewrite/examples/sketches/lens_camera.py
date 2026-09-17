"""A zoom camera with depth of field: cones pulled through non-rigid maps, in flat or spherical space.

An ideal thin lens is a projective collineation of space, P ↦ P + centre (P ∨ plane) / f, and
its action on lines, L ↦ L - (centre ∨ (L ∧ plane)) / f, is the join of the images: the train
composes either way. The rays a scene point sends through the aperture form a cone, the
pullback of a ball through the central projection from the point onto the aperture plane.
The train carries that cone to the image cone, a pullback through the inverse of its point
map, and the sensor cuts the image cone in the point's blur conic. Nothing asks where the
point focuses; the cone's vertex is wherever the collineation put it.

The core logic never names the metric: with w² = 1 instead of 0 the same lens maps, quadrics,
pullbacks and checks run on the 3-sphere, where translators are rotations toward the pole and
a ball of "radius" r is a cap of angular radius atan r. Only the point constructor, which
motors a chosen origin, and the chart readouts for drawing know which space they are in.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import NumpyContext
from numga.algebras import PGA3D
from numga.gatype.traits import Versor

from examples import PLOT_DIR
from examples.animation import capture, save_gif

ga = PGA3D
ctx = NumpyContext(ga)
mv = ctx.multivector
P = ga.subspace.antivector()
Point = ga.gatype.antivector()
Plane = ga.gatype.vector()
SensorPoint = ga.gatype.from_blades("zxw xyw zyx")
SensorPlane = ga.gatype.from_blades("y z w")
Line = ga.gatype.bivector()
Motor = ga.gatype.rotor()
PointMap = ga.gatype((Point, Point))          # collineation: point <= point
Quadric = ga.gatype((Plane, Point))           # primal quadric: polar plane <= point
DualQuadric = ga.gatype((Point, Plane))       # dual quadric: pole <= plane
Camera = ga.gatype((Point, Point, Point))     # sensor point <= (scene point, pupil point)
SENSOR = (0.225, 0.175)                       # half extents of the sensor window in its own frame


# --- plumbing -------------------------------------------------------------------------
def unit(points: Point) -> Point:
    """The representative of each point with positive unit weight; a unit point is a versor."""
    return (points / (mv.w & points)).normalized().with_traits(Versor)


origin = unit(mv.zyx)


def point(coords: np.ndarray) -> Point:
    """The origin carried by the translator with the given displacement."""
    return ((mv.xw * coords[..., 0] + mv.yw * coords[..., 1] + mv.zw * coords[..., 2]) * 0.5).exp() >> origin


def direction(coords: np.ndarray) -> Point:
    """An ideal point: a direction."""
    return mv.antivector(np.concatenate([coords, np.zeros_like(coords[..., :1])], axis=-1))


def sensor_yz(frame: Motor, points: Point) -> np.ndarray:
    """(y, z) coordinates of sensor points, read in the sensor's own frame."""
    k = (frame << points).cast(P).kernel
    return k[..., 1:3] / k[..., 3:]


def render(frame: Motor, cones: Quadric, layer: np.ndarray, energy: float, resolution: tuple[int, int] = (280, 360), supersample: int = 2) -> np.ndarray:
    """Rasterise the sensor from the image cones as implicit functions.

    A pixel's coverage by a point's blur disc comes from a first-order signed distance to the
    disc boundary: the cone's form at the pixel over the length of its gradient, which is twice
    the Euclidean normal of the polar plane. The edge is a logistic two pixels wide, a crude
    diffraction limit that spreads a focused point over a few pixels. Each disc deposits the
    same total energy, scaled by the aperture area, so a point in focus is a bright dot and a
    defocused one a dim wide disc. Scene points of each depth layer light one colour channel.
    """
    rows, cols = (resolution[0] * supersample, resolution[1] * supersample)
    y, z = np.meshgrid(np.linspace(-SENSOR[0], SENSOR[0], cols), np.linspace(SENSOR[1], -SENSOR[1], rows))
    pixels = frame >> point(np.stack([np.zeros_like(y), y, z], axis=-1).reshape(-1, 3))
    polar = cones.reshape(-1, 1)(pixels)
    with np.errstate(divide="ignore", invalid="ignore"):
        distance = (pixels & polar).kernel[..., 0] / (2 * polar.norm().kernel[..., 0])
    coverage = 1.0 / (1.0 + np.exp(-distance / (2 * SENSOR[0] / cols * 2)))   # logistic edge, two pixels wide
    share = coverage / np.maximum(coverage.sum(axis=-1, keepdims=True), 1e-12) * energy
    image = np.zeros((rows * cols, 3))
    for channel in range(3):
        image[:, channel] = share[layer.ravel() == channel].sum(axis=0)
    image = image.reshape(resolution[0], supersample, resolution[1], supersample, 3).mean(axis=(1, 3))
    return np.clip(image * 550.0, 0.0, 1.0)


def draw_side(ax, planes: list[Plane], heights: list[float], legs: list[Point], scene: Point) -> None:
    """Side view in the x-y plane: element planes as segments, the scene layers, and a ray fan drawn leg by leg."""
    ax.cla()
    for plane, height in zip(planes, heights):
        top, bottom = xyz(plane ^ mv.z ^ (mv.y - mv.w * height)), xyz(plane ^ mv.z ^ (mv.y + mv.w * height))
        ax.plot([top[0], bottom[0]], [top[1], bottom[1]], color="gray", linewidth=2)
    xy = xyz(scene)[..., :2].reshape(-1, 2)
    ax.scatter(xy[:, 0], xy[:, 1], s=4, color="tab:gray")
    for a, b in zip(legs[:-1], legs[1:]):
        for p, q in zip(xyz(a).reshape(-1, 3), xyz(b).reshape(-1, 3)):
            ax.plot([p[0], q[0]], [p[1], q[1]], color="tab:orange", linewidth=0.7)
    ax.set_xlim(-3.6, 2.6); ax.set_ylim(-1.2, 1.2); ax.set_aspect("equal"); ax.axis("off")


def xyz(points: Point) -> np.ndarray:
    k = points.cast(P).kernel
    return k[..., :3] / k[..., 3:]


def camera_settings():
    """Construct placements, focus points, tilts and pupil samples before tracing."""
    static = [("wide, aperture 0.45, focused at 2.2", 1.4, -2.2, 0.0, .45),
              ("tele, aperture 0.45, focused at 2.2", 1.8, -2.2, 0.0, .45),
              ("tele, aperture 0.45, sensor tilted 25°", 1.8, -2.2, np.radians(25), .45)]
    motion = [(f"rear lens at {1.6 + .2*np.sin(t):.2f}, focused at {2.4 - .8*np.cos(t):.2f}, aperture radius {.3 + .1*np.sin(2*t):.2f}",
               1.6 + .2*np.sin(t), -2.4 + .8*np.cos(t), 0.0, .3 + .1*np.sin(2*t))
              for t in np.linspace(0.0, 2*np.pi, 72, endpoint=False)]
    return [(title, (mv.xw * (rear_at / 2)).exp(), point(np.array([focus_at, 0., 0.])),
             (mv.xy * (tilt / 2)).exp(), radius,
             point(np.array([[0., radius, 0.], [0., 0., 0.], [0., -radius, 0.]])))
            for title, rear_at, focus_at, tilt, radius in static + motion]


def draw_camera(states, layer, scene: Point, centre: Point, plot_path: str, animation_path: str):
    """Consume camera geometry for the three still views and optional animation."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), dpi=120)
    results = []
    for ax in axes:
        title, collineation, cam, frame, cones, radius, planes, legs = next(states)
        ax.imshow(render(frame, cones, layer, 1.0), interpolation="nearest")
        ax.set_title(title); ax.axis("off")
        results.append((collineation, cam, frame, cones))
        print(f"{title}: camera matrix\n{np.round(cam.bind({1: centre}).kernel, 3)}")
    if plot_path:
        fig.savefig(plot_path, bbox_inches="tight")
        print(f"Figure saved to {plot_path}")
    if animation_path:
        anim = plt.figure(figsize=(6, 6.6), dpi=100)
        top, side = anim.subplots(2, 1, height_ratios=(3, 1))
        anim.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02, hspace=0.08)
        frames_out = []
        for title, collineation, cam, frame, cones, radius, planes, legs in states:
            top.cla()
            top.imshow(render(frame, cones, layer, (radius / .45)**2), interpolation="nearest")
            top.set_title(title, fontsize=9); top.axis("off")
            draw_side(side, planes, [radius, .6, .35], legs, scene)
            frames_out.append(capture(anim))
        plt.close(anim)
        save_gif(frames_out, animation_path, duration_ms=60, colors=128)
    return fig, results


# --- math -----------------------------------------------------------------------------
def main(
    plot_path: str = str(PLOT_DIR / "sketch_lens_camera.png"),
    animation_path: str = str(PLOT_DIR / "sketch_lens_camera.gif"),
) -> plt.Figure:
    dx, dy, dz = direction(np.eye(3))
    axis = origin & dx                                            # the optical axis
    home = mv.x                                                   # every element lives in the plane x = 0

    # Elements in the home plane. A thin lens as a collineation of points and as the induced
    # map on lines; a ball of the aperture radius, whose section with the home plane is the
    # aperture rim; and, for the checks only, the aperture disc as a flat dual quadric.
    front_points = Point + origin * (home & Point) / 1.0
    rear_points = Point + origin * (home & Point) / 0.6
    front_lines = Line - (origin & (Line ^ home)) / 1.0
    rear_lines = Line - (origin & (Line ^ home)) / 0.6
    unit_ball = mv.x * (mv.x & Point) + mv.y * (mv.y & Point) + mv.z * (mv.z & Point) - mv.w * (mv.w & Point)
    unit_disc = dy * (dy & Plane) + dz * (dz & Plane) - origin * (origin & Plane)

    # A scene: a grid of points at three depths in front of the camera.
    depth = np.array([-3.2, -2.2, -1.6])
    grid = np.stack(np.meshgrid(depth, np.linspace(-0.8, 0.8, 5), np.linspace(-0.5, 0.5, 4), indexing="ij"), axis=-1)
    scene = point(grid)

    # Prepare the camera motion as GA inputs; every setting uses the same optical train.
    layer = np.arange(3)[:, None, None].repeat(5, 1).repeat(4, 2)
    place_front = (mv.xw * 0.5).exp()
    settings = camera_settings()

    def scenes():
        for title, place_rear, focus, tilt, radius, rim in settings:
            # Place the lenses, then compose their point maps and their line maps.
            front = place_front >> front_lines(place_front << Line)
            rear = place_rear >> rear_lines(place_rear << Line)
            front_plane, rear_plane = place_front >> home, place_rear >> home
            collineation = (place_rear >> rear_points(place_rear << Point))(place_front >> front_points(place_front << Point))
            train = rear(front)
            ball = unit_ball + mv.w * (mv.w & Point) * (1.0 - radius**2)
            pupil_ball = place_front >> ball(place_front << Point)

            # The focus point's image places the sensor; tilt turns it about that image.
            image = unit(collineation(focus))
            frame = (image / origin).square_root() * tilt
            sensor = frame >> home
            cam = train(Point & Point) ^ sensor

            # Project through each subject onto the pupil, then pull back its quadric.
            # Pull back once more through the inverse lens train to get the image cones.
            project = (scene & Point) ^ front_plane
            cone = project.transpose()(Plane.dual()).dual_inverse()(pupil_ball(project))
            back = collineation.inverse()
            cones = back.transpose()(Plane.dual()).dual_inverse()(cone(back))

            subject = scene[0, 2, 1]
            rays = subject & (place_front >> rim)
            legs = [subject.reshape(1).broadcast_to((3,)), rays ^ front_plane,
                    front(rays) ^ rear_plane, rear(front(rays)) ^ sensor]
            yield (title, collineation, cam, frame, cones, radius,
                   [front_plane, rear_plane, sensor], legs)

    fig, results = draw_camera(scenes(), layer, scene, place_front >> origin, plot_path, animation_path)

    # --- checks: kernel-level assertions, deliberately outside the demonstration ----------
    def section(cone: Quadric, start: Point, frame: Motor, samples: int = 48) -> Point:
        """Boundary of the sensor's section of a cone, traced from a point inside it along the sensor plane."""
        theta = np.linspace(0.0, 2 * np.pi, samples)
        across = frame >> direction(np.stack([np.zeros(samples), np.cos(theta), np.sin(theta)], axis=-1))
        a, b, c = across & cone(across), across & cone(start), start & cone(start)
        discriminant = np.maximum((b * b - a * c).kernel[..., 0], 0.0)     # rounding-level negative at the vertex
        return start + across * ((np.sqrt(discriminant) - b.kernel[..., 0]) / a.kernel[..., 0])

    collineation, cam, frame, cones = results[2]
    sensor, centre = frame >> home, place_front >> origin
    pupil = place_front >> (unit_disc * 0.45**2 - origin * (origin & Plane) * (1.0 - 0.45**2))(place_front << Plane)
    # The train on lines is the join of the collineation's images: the lens is a collineation.
    Q = point(np.array([[-1.0, 0.3, 0.1], [-1.5, 0.2, -0.4], [-2.0, -0.5, 0.6]]))
    np.testing.assert_allclose(cam.bind({1: Q})(scene[0, 0, 0]).kernel, ((collineation(scene[0, 0, 0]) & collineation(Q)) ^ sensor).kernel, atol=1e-12)
    # The image cone's vertex is the collineation's image of the subject.
    np.testing.assert_allclose((collineation(scene) & cones(collineation(scene))).kernel, 0.0, atol=1e-9)
    # The sensor's section of the image cone, traced from the chief-ray hit, lies on the conic
    # obtained the short way: the aperture disc, a flat dual quadric, pushed through the
    # pupil-to-sensor collineation of the same point.
    for index in ((0, 0, 0), (2, 4, 3)):
        start = (collineation(scene[index]) & collineation(centre)) ^ sensor
        boundary = section(cones[index], start, frame)
        to_sensor = cam.bind({0: scene[index]})
        pushed = to_sensor(pupil(to_sensor.transpose()(Plane.dual()).dual_inverse()))
        section_dual = (frame << pushed(frame >> SensorPlane)).cast(SensorPoint.output_subspace)
        hits = (frame << boundary).cast(SensorPoint.output_subspace)
        np.testing.assert_allclose((hits & section_dual.solve(hits)).kernel, 0.0, atol=1e-10)
    # A point at the focus depth images to a point: its section collapses onto the chief-ray hit.
    collineation, cam, frame, cones = results[0]
    start = (collineation(scene[1, 2, 1]) & collineation(centre)) ^ (frame >> home)
    np.testing.assert_allclose(sensor_yz(frame, section(cones[1, 2, 1], start, frame)), sensor_yz(frame, start.reshape(1).broadcast_to((48,))), atol=1e-9)
    return fig


if __name__ == "__main__":
    main()
