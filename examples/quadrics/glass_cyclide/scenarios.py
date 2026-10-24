"""A glass Dupin cyclide with a bite taken out of it, a ball behind it, on a checkered floor, orbited."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.quadrics.glass_cyclide import core
from examples.quadrics.glass_cyclide.core import Quadric, ball, cyclide, infinity, moved, mv, origin, sphere, translator

# The torus the cyclide comes from, and the sphere it is inverted in: off the torus's centre, so the tube
# swells on the far side.
torus_radius, torus_tube = 2.5, 0.9
inversion = (translator(mv.x * 8.0) >> sphere(origin, 6.25)).normalized()
# Where the inverted torus lies, and how it is stood up above the floor: the centre of its bounding
# sphere carried by `lift`, after a tilt about x.
cyclide_centre = mv.x * 2.04
cyclide_radius = 2.6
cyclide_tilt = np.radians(55.0)
lift = mv.z * 2.25
# A ball taken out of the cyclide's thick side, and a ball resting on the floor behind it.
bite_centre = translator(mv.x * 0.6 + mv.y * 1.2 + mv.z * 0.9) >> origin
bite_radius = 0.9
ball_centre = translator(mv.x * 0.6 + mv.y * 3.4 + mv.z * 1.1) >> origin
ball_radius = 1.1
floor_radius = 60.0
# Each solid's colour or glass tint, transparency and refractive index, in the order of `solids`:
# cyclide, ball, floor.
colours = np.array([[0.9, 0.96, 0.97], [0.85, 0.22, 0.16], [0.88, 0.87, 0.85]])
transparencies = np.array([0.96, 0.0, 0.0])
refractions = np.array([1.5, 1.0, 1.0])
tiles = 0.8
lamp = (mv.x * 0.5 - mv.y * 0.6 + mv.z * 0.9).normalized()
# The camera orbits the cyclide.
target = translator(mv.z * 1.9) >> origin
distance = 11.0
elevation = np.radians(20.0)
half_view = 0.31
image_shape = (450, 720)
animation_shape = (330, 528)
n_frames = 48
duration_ms = 80


# --- math -----------------------------------------------------------------------------
def placement() -> core.Motor:
    """The cyclide's motor: its bounding centre to the origin, tilted about x, lifted above the floor."""
    return translator(lift) * (mv.yz * (-cyclide_tilt / 2)).exp() * translator(-cyclide_centre)


def solids() -> tuple[Quadric, dict]:
    """Every solid's quadrics, solid after solid, and what the shader reads of each solid."""
    place = placement()
    shapes = [
        # The glass cyclide, less the bite: outside the bite's ball, the ball negated.
        [moved(cyclide(torus_radius, torus_tube, inversion), place), -ball(place >> sphere(bite_centre, bite_radius))],
        [ball(sphere(ball_centre, ball_radius))],
        # The floor: everything below the plane z = 0.
        [ball(-mv.z)],
    ]
    counts = np.array([len(shape) for shape in shapes])
    ends = np.cumsum(counts)
    quadrics = stack([quadric.cast(Quadric) for shape in shapes for quadric in shape])      # [quadrics] Quadric
    centres = stack([translator(lift) >> origin, ball_centre, origin])                      # [solids] Point
    return quadrics, dict(
        first=ends - counts, end=ends, centre=centres, radius=np.array([cyclide_radius, ball_radius * 1.01, floor_radius]),
        colour=colours, transparency=transparencies, refraction=refractions,
        across=stack([0 * mv.x, 0 * mv.x, mv.x * tiles]), along=stack([0 * mv.y, 0 * mv.y, mv.y * tiles]))


def cameras(n_cameras: int) -> tuple[core.Rotation, core.Point]:
    """Cameras orbiting the target, each a turn from one looking along -z with y up, and an eye."""
    azimuths = np.linspace(0, 2 * np.pi, n_cameras, endpoint=False)
    pitch = (mv.yz * (-(np.pi / 2 - elevation) / 2)).exp()
    turns = (mv.xy * (-azimuths / 2)).exp() * pitch                                       # [cameras] Rotation
    return turns, translator((turns >> mv.z) * distance) >> target                        # [cameras] Rotation, [cameras] Point


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.quadrics.glass_cyclide import render
    from examples.util.wgsl import image_figure

    tracer = render.shader()
    quadrics, solid_values = solids()
    turns, eyes = cameras(1)
    still, = render.frames(tracer, quadrics, solid_values, turns, eyes, lamp, half_view, image_shape)
    save_figure(image_figure(still), "glass_cyclide")
    turns, eyes = cameras(n_frames)
    save_animation(render.frames(tracer, quadrics, solid_values, turns, eyes, lamp, half_view, animation_shape),
                   "glass_cyclide", duration_ms)

    # --- checks
    # The torus's points, inverted in the sphere, lie on the cyclide.
    points = inversion >> core.torus_points(torus_radius, torus_tube, 24, 12)              # [across, around] Point
    # The form scales with the square of a point's weight, its pairing with the point at infinity.
    form = (points & cyclide(torus_radius, torus_tube, inversion)(points)) / (infinity & points).squared()
    np.testing.assert_allclose(form.kernel, 0, atol=1e-10)


if __name__ == "__main__":
    main()
