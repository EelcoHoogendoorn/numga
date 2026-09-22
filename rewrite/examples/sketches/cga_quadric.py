"""CGA cyclides and circle vortices, with spheres/planes as 1-vectors.

Run with --scene linked_vortex --frames 40 for the interlocking animation.
The six_families scene uses Example 7 of https://arxiv.org/abs/1106.1354;
hyperboloid and two_lobed use our own parameters in the paper's families.
"""

from collections.abc import Iterator

import numpy as np

from numga import Extensor

from examples.sketches.cga_quadric_plumbing import (
    Direction, Point, shade, nearest_depth,
    infinity, mv, origin,
)

from examples.sketches.cga_quadric_scenarios import Scene


def camera(scene: Scene) -> tuple[Extensor, Extensor, Direction]:
    """Affine camera-to-world point map, direction map, and sensor pixels."""
    forward = (scene.target - scene.position).normalized()
    right = ((forward ^ mv.z) * mv.xyz.inverse()).normalized()
    up = (right ^ forward) * mv.xyz.inverse()
    height, width = scene.shape[0] * scene.supersample, scene.shape[1] * scene.supersample
    u = (2 * (np.arange(width) + 0.5) / width - 1)[None, :]
    v = (1 - 2 * (np.arange(height) + 0.5) / height)[:, None] * height / width
    pixels = (-mv.x + (mv.y * u + mv.z * v) * np.tan(scene.fov / 2)).reshape(-1)
    screen = -forward * (mv.x | Direction) + right * (mv.y | Direction) + up * (mv.z | Direction)
    aim = (1 - forward * mv.x).normalized()
    roll = (1 + right * (aim >> mv.y)).normalized()
    pose = (-0.5 * (scene.position ^ infinity)).exp() * roll * aim
    return pose >> Point, screen, pixels


def render(scene: Scene) -> Iterator[np.ndarray]:
    # Camera geometry and light directions in its frame.
    metric = Direction | Direction      # the Euclidean metric on directions: raises a gradient form to a direction
    camera_map, screen, pixels = camera(scene)
    main_light = camera_map & scene.main_light_source
    fill_light = camera_map & scene.fill_light_source
    main_weight = -(scene.main_light_source | infinity)
    fill_weight = -(scene.fill_light_source | infinity)
    up = screen.inverse()(mv.z)

    # A direction slot generates translations along any ray in camera coordinates.
    ray_translation = -0.5 * (Direction ^ infinity)

    # In camera coordinates a ray point is origin + t linear(pixel) + t² quadratic(pixel, pixel).
    ray_linear = ray_translation.commutator(origin) * 2
    ray_quadratic = ray_translation.commutator(ray_linear)
    world_ray_linear = camera_map(ray_linear)

    for motors in scene.motors:
        world_quadrics = motors >> scene.surfaces(motors << Point)
        # Pull the world quadrics back through the affine camera map once per body.
        bodies = camera_map & world_quadrics(camera_map)
        constant = bodies(origin, origin)
        linear = bodies(origin, ray_linear) * 2
        quadratic = bodies(ray_linear, ray_linear) + bodies(origin, ray_quadratic) * 2
        # The affine camera fixes the ideal point in ray_quadratic. Contract it
        # before the camera pullback to preserve exact zeros in lower-degree surfaces.
        quadratic_polar = world_quadrics(ray_quadratic)
        cubic = (world_ray_linear & quadratic_polar) * 2
        quartic = ray_quadratic & quadratic_polar
        def traces():
            # Dense ray batches bound the quartic solver's working memory.
            for start in range(0, pixels.shape[0], 8192):
                pixel = pixels[start:start + 8192]
                depth = np.zeros(pixel.shape)
                body_idx = np.zeros(pixel.shape, dtype=int)
                for body in range(bodies.shape[0]):
                    candidate = nearest_depth(
                        constant[body],
                        linear[body](pixel),
                        quadratic[body](pixel, pixel),
                        cubic[body](pixel, pixel, pixel),
                        quartic[body](pixel, pixel, pixel, pixel),
                    )
                    body_idx = np.where(candidate > depth, body, body_idx)
                    depth = np.maximum(depth, candidate)
                visible = depth > 0
                t = 1 / np.where(visible, depth, 1)
                hit = origin + ray_linear(pixel) * t + ray_quadratic(pixel, pixel) * t * t

                # Gather the winning polarity and shade all pixels together.
                # Bind the rightmost hit first, reducing the form before the derivative.
                tangent = ray_translation.commutator(hit) * 2
                derivative = bodies[body_idx](tangent, hit)
                normal = metric.solve(derivative).normalized()
                # A sphere's incidence gradient points away from its centre;
                # a plane gives a constant direction. Both light types use this map.
                main_direction = -metric.solve(main_light(tangent)).normalized()
                fill_direction = -metric.solve(fill_light(tangent)).normalized()
                main_strength = (1 + 2 * main_weight * main_light(hit)).inverse()
                fill_strength = (1 + 2 * fill_weight * fill_light(hit)).inverse()
                yield (normal, pixel.normalized(), visible, scene.colors[body_idx],
                       main_direction, fill_direction, main_strength, fill_strength)
        yield shade(traces(), up, scene.shape, scene.supersample)


if __name__ == "__main__":
    from examples.sketches.cga_quadric_plumbing import arguments
    from examples.sketches.cga_quadric_scenarios import main
    main(**arguments())
