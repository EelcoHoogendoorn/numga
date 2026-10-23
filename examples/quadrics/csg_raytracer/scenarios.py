"""A ruled hyperboloid, a hollow bowl cut open, and a mirror sphere on a checkered floor, orbited."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.quadrics.csg_raytracer import core
from examples.quadrics.csg_raytracer.core import Point, Quadric, ellipsoid, half_space, hyperboloid, mv, placed, translator

# The hyperboloid's waist radius and its half-height, where two planes cut it.
waist_axes = np.array([0.55, 0.55, 0.85])
hyperboloid_place = mv.direction([-1.2, 0.4, 0.85])
hyperboloid_tilt = np.radians(12.0)
# The bowl: an ellipsoid less a smaller one, cut by a plane tilted off its top.
bowl_axes = np.array([0.75, 0.75, 0.75])
wall_fraction = 0.8
bowl_place = mv.direction([1.1, -0.2, 0.75])
bowl_tilts = np.radians([35.0, 15.0])
bowl_cut = (mv.y * 0.15 + mv.z * 0.98) - mv.w * 0.1
mirror_radius = 0.55
mirror_place = mv.direction([0.0, 0.1, 0.55])
# Each solid's colour and reflectivity, in the order of `solids`: hyperboloid, bowl, mirror, floor.
colours = np.array([[0.84, 0.48, 0.24], [0.15, 0.45, 0.55], [0.95, 0.95, 0.96], [0.88, 0.87, 0.85]])
reflectivities = np.array([0.0, 0.0, 0.88, 0.0])
# Unit steps of each solid's checker: stripes crossing the hyperboloid's rulings, tiles on the floor.
stripes = 8.0
tiles = 1.4
# The lamp's direction, and the camera orbiting a point above the floor.
lamp = mv.direction([0.5, -0.5, 0.7])
target = mv.point([0.0, 0.2, 0.7])
distance = 4.6
elevation = np.radians(28.0)
half_view = 0.42
image_shape = (450, 720)
animation_shape = (330, 528)
n_frames = 48
duration_ms = 80
n_rulings = 12


# --- math -----------------------------------------------------------------------------
def solids() -> tuple[Quadric, dict]:
    """Every solid's quadrics, solid after solid, and each solid's range of them, colour, reflectivity and
    checker planes."""
    lift = translator(hyperboloid_place) * (mv.xz * (-hyperboloid_tilt / 2)).exp()                 # [] Motor
    turn_bowl = translator(bowl_place) * (mv.yz * (-bowl_tilts[0] / 2)).exp() * (mv.xz * (-bowl_tilts[1] / 2)).exp()  # [] Motor
    height = waist_axes[2]
    shapes = [
        # The hyperboloid between the planes a half-height above and below its waist.
        [placed(hyperboloid(waist_axes), lift),
         placed(half_space(mv.z - mv.w * height), lift),
         placed(half_space(-mv.z - mv.w * height), lift)],
        # The bowl: inside the outer ellipsoid, outside the inner one, below the cut.
        [placed(ellipsoid(bowl_axes), turn_bowl),
         -placed(ellipsoid(bowl_axes * wall_fraction), turn_bowl),
         placed(half_space(bowl_cut / (bowl_cut | bowl_cut).square_root()), turn_bowl)],
        [placed(ellipsoid(np.full(3, mirror_radius)), translator(mirror_place))],
        # The floor: everything below the plane z = 0.
        [half_space(mv.z).cast(Quadric)],
    ]
    counts = np.array([len(shape) for shape in shapes])
    ends = np.cumsum(counts)
    quadrics = stack([quadric for shape in shapes for quadric in shape])          # [quadrics] Quadric
    across = stack([lift >> (mv.x + mv.z) * stripes, 0 * mv.x, 0 * mv.x, mv.x * tiles])      # [solids] Plane
    along = stack([lift >> (mv.y - mv.z) * stripes, 0 * mv.y, 0 * mv.y, mv.y * tiles])       # [solids] Plane
    return quadrics, dict(first=ends - counts, end=ends, colour=colours, reflectivity=reflectivities,
                          across=across, along=along)


def orbit(n_cameras: int) -> core.Motor:
    """Cameras around the target, starting in front of the scene."""
    return core.orbit(target, distance, elevation, np.linspace(0, 2 * np.pi, n_cameras, endpoint=False))


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.quadrics.csg_raytracer import render
    from examples.util.wgsl import image_figure

    tracer = render.shader()
    quadrics, solid_values = solids()
    still, = render.frames(tracer, quadrics, solid_values, orbit(1), lamp, half_view, image_shape)
    save_figure(image_figure(still), "csg_raytracer")
    save_animation(render.frames(tracer, quadrics, solid_values, orbit(n_frames), lamp, half_view, animation_shape),
                   "csg_raytracer", duration_ms)

    # --- checks
    # Both families of straight lines through the hyperboloid's waist lie in its surface.
    waist, rising, falling = core.ruling_lines(waist_axes, n_rulings)
    reach = np.linspace(-2, 2, 5)[:, None]
    for direction in (rising, falling):
        np.testing.assert_allclose(core.evaluate(hyperboloid(waist_axes), waist + direction * reach).kernel, 0, atol=1e-12)
    # Every camera looks at the target.
    cameras = orbit(n_frames)
    eyes = cameras >> core.origin
    looks = cameras >> -mv.z.dual()
    np.testing.assert_allclose((eyes + looks * distance - target).kernel, 0, atol=1e-12)


if __name__ == "__main__":
    main()
