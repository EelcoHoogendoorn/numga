"""Scene setup and animation entry point for ray-traced composite quadric solids."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import concatenate, stack
from examples.quadrics.csg_raytracer import core
from examples.quadrics.csg_raytracer.core import Motor, Quadric, ellipsoid, half_space, hyperboloid, mv

# Waist radius and half-height of the ruled hyperboloid between its two end planes.
waist_axes = np.array([0.55, 0.55, 0.85])
hyperboloid_place = mv.direction([-1.2, 0.4, 0.85])                              # [] Point-
hyperboloid_tilt = np.radians(12.0)
# Hollow bowl cut from two concentric spheres and sliced open across the top.
bowl_axes = np.array([0.75, 0.75, 0.75])
wall_fraction = 0.8
bowl_place = mv.direction([1.1, -0.2, 0.75])                                     # [] Point-
bowl_tilts = np.radians([35.0, 15.0])
bowl_cut = (mv.y * 0.15 + mv.z * 0.98) - mv.w * 0.1                              # [] Plane+
mirror_radius = 0.55
mirror_place = mv.direction([0.0, 0.1, 0.55])                                    # [] Point-
# Material colours and mirror reflectivities for hyperboloid, bowl, mirror sphere, and floor.
colours = np.array([[0.84, 0.48, 0.24], [0.15, 0.45, 0.55], [0.95, 0.95, 0.96], [0.88, 0.87, 0.85]])
reflectivities = np.array([0.0, 0.0, 0.88, 0.0])
# Spatial frequencies for the hyperboloid's checker stripes and the floor's tiles.
stripe_rate = 8.0
tile_rate = 1.4
# Key-lamp direction and camera orbit parameters around a target above the floor.
lamp = mv.direction([0.5, -0.5, 0.7])                                            # [] Point-
target = mv.point([0.0, 0.2, 0.7])                                               # [] Point-
distance = 4.6
elevation = np.radians(28.0)
half_view = 0.42
image_shape = (450, 720)
animation_shape = (330, 528)
n_frames = 96
frame_duration_ms = 40
n_rulings = 12


# --- math -----------------------------------------------------------------------------
def scene_solids() -> tuple[Quadric, dict]:
    """Places the hyperboloid, hollow bowl, mirror sphere, and floor quadrics (`[8] Plane+ <- Point-`)
    alongside their per-solid range, material, and checker-plane tables."""
    # Rigid motors translate each solid from the origin along its placement direction after tilting.
    hyperboloid_motor = (((mv.w ^ hyperboloid_place.dual()) / 2).exp()
                         * (mv.xz * (-hyperboloid_tilt / 2)).exp())              # [] Motor+
    bowl_shift = ((mv.w ^ bowl_place.dual()) / 2).exp()                          # [] Motor+
    bowl_tilt = ((mv.yz * (-bowl_tilts[0] / 2)).exp()
                 * (mv.xz * (-bowl_tilts[1] / 2)).exp())                         # [] Motor+
    bowl_motor = bowl_shift * bowl_tilt                                          # [] Motor+
    mirror_motor = ((mv.w ^ mirror_place.dual()) / 2).exp()                      # [] Motor+
    height = waist_axes[2]
    # Cap the ruled hyperboloid between horizontal half-spaces one half-height above and below its waist.
    hyperboloid_parts = stack([hyperboloid(waist_axes), half_space(mv.z - mv.w * height),
                               half_space(-mv.z - mv.w * height)])               # [3] Plane+ <- Point-
    # Subtract the inner sphere by negation and slice off the top with the tilted half-space.
    bowl_parts = stack([ellipsoid(bowl_axes), -ellipsoid(bowl_axes * wall_fraction),
                        half_space(bowl_cut.normalized())])                      # [3] Plane+ <- Point-
    mirror_parts = ellipsoid(np.full(3, mirror_radius)).reshape(1)               # [1] Plane+ <- Point-
    floor_parts = half_space(mv.z).cast(Quadric).reshape(1)                      # [1] Plane+ <- Point-
    # Pull back each solid's input points and push forward its output planes by the solid's motor.
    placed_hyperboloid = hyperboloid_motor >> hyperboloid_parts(hyperboloid_motor << core.Point)
    placed_bowl = bowl_motor >> bowl_parts(bowl_motor << core.Point)
    placed_mirror = mirror_motor >> mirror_parts(mirror_motor << core.Point)
    shapes = [placed_hyperboloid, placed_bowl, placed_mirror, floor_parts]
    counts = np.array([shape.shape[0] for shape in shapes])
    ends = np.cumsum(counts)
    quadrics = concatenate(shapes)                                               # [8] Plane+ <- Point-
    first_stripe = stack([hyperboloid_motor >> ((mv.x + mv.z) * stripe_rate),
                          0 * mv.x, 0 * mv.x, mv.x * tile_rate])                 # [4] Plane+
    second_stripe = stack([hyperboloid_motor >> ((mv.y - mv.z) * stripe_rate),
                           0 * mv.y, 0 * mv.y, mv.y * tile_rate])                # [4] Plane+
    return quadrics, dict(first=ends - counts, end=ends, colour=colours, reflectivity=reflectivities,
                          first_stripe=first_stripe, second_stripe=second_stripe)


def orbit_cameras(n_cameras: int) -> Iterator[Motor]:
    """Yields `n_cameras` camera pose motors (`[] Motor+`) spaced evenly in azimuth around `target`."""
    for azimuth in np.linspace(0, 2 * np.pi, n_cameras, endpoint=False):
        yield core.camera_orbit(target, distance, elevation, azimuth)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    """Renders the front-view figure and full-orbit animation and checks rulings and camera aim."""
    from examples.quadrics.csg_raytracer import render
    from examples.util.animation import save_animation, save_figure
    from examples.util.panels import image_figure

    tracer = render.shader()
    quadrics, solids = scene_solids()
    still, = render.frames(tracer, quadrics, solids, orbit_cameras(1), lamp, half_view, image_shape)
    save_figure(image_figure(still), "csg_raytracer")

    kept_cameras = []

    def streamed_cameras() -> Iterator[Motor]:
        for camera in orbit_cameras(n_frames):
            kept_cameras.append(camera)
            yield camera

    save_animation(render.frames(tracer, quadrics, solids, streamed_cameras(), lamp, half_view, animation_shape),
                   "csg_raytracer", frame_duration_ms)
    cameras = stack(kept_cameras)                                                # [n_frames] Motor+

    # checks
    # Both straight-line ruling families through the hyperboloid's waist lie identically on its surface.
    waist, rising, falling = core.ruling_lines(waist_axes, n_rulings)            # [n_rulings] Point-, Point-, Point-
    n_offsets = 5
    offsets = np.linspace(-2, 2, n_offsets)[:, None]
    surface = hyperboloid(waist_axes)                                            # [] Plane+ <- Point-
    ruling_points = waist + stack([rising, falling])[:, None] * offsets          # [2, n_offsets, n_rulings] Point-
    assert ((surface(ruling_points) & ruling_points).abs() < 1e-10).all()
    # Every streamed camera sits `distance` from `target` and aims straight at it.
    eyes = cameras >> core.origin                                                # [n_frames] Point-
    aim = cameras >> (-mv.z.dual())                                              # [n_frames] Point-
    assert ((eyes + aim * distance - target).antinorm() < 1e-10).all()


if __name__ == "__main__":
    main()
