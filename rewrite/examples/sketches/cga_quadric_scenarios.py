"""CGA quadric scenes: construct only the selected surface and its circle motion."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from pathlib import Path

from numga import Extensor
from examples import PLOT_DIR
from examples.sketches.cga_quadric_plumbing import (
    Direction, Point, Quadric, Sphere, infinity, mv, zero_sphere,
)


def torus(radius: float, tube: float) -> Quadric:
    sphere = zero_sphere - infinity * ((radius * radius + tube * tube) / 2)
    return sphere * (sphere & Point) + radius * radius * (
        mv.z * (mv.z & Point) - tube * tube * infinity * (infinity & Point)
    )


def cyclide():
    surface = torus(1.4, 0.5)
    # An off-centre sphere inversion gives the torus unequal tube widths.
    shift = (-0.5 * ((mv.x * 3.2) ^ infinity)).exp()
    inversion = (shift >> (zero_sphere - infinity * 4.5)).normalized()
    pose = (mv.xy * 0.12).exp() * (mv.yz * -0.22).exp()
    placement = pose * inversion
    return ((placement >> surface(placement << Point), mv.bivector()),)


def peanut():
    a, b = 1.25, 1.31
    sphere = zero_sphere - infinity * (a * a / 2)
    surface = sphere * (sphere & Point) + a * a * (
        mv.y * (mv.y & Point) + mv.z * (mv.z & Point)
    ) - (b**4 / 4) * infinity * (infinity & Point)
    pose = (mv.xy * 0.10).exp()
    return ((pose >> surface(pose << Point), mv.bivector()),)


def elliptic_ring():
    surface = torus(1.4, 0.5) + (0.10 * 1.4**2) * mv.y * (mv.y & Point)
    pose = (mv.yz * -0.18).exp()
    return ((pose >> surface(pose << Point), mv.bivector()),)


def split_ring():
    surface = torus(1.4, 0.5) + (0.32 * 1.4**2) * mv.y * (mv.y & Point)
    pose = (mv.yz * -0.18).exp()
    return ((pose >> surface(pose << Point), mv.bivector()),)


def pinched_surface() -> Quadric:
    # Inversion closes the hyperboloid's infinite ends at a singular point.
    surface = (mv.z * (mv.z & Point) - mv.x * (mv.x & Point)
               - mv.y * (mv.y & Point) + infinity * (infinity & Point))
    inversion = (zero_sphere - infinity).normalized()
    return inversion >> surface(inversion << Point)


def pinched():
    surface = pinched_surface()
    pose = (mv.yz * -0.18).exp()
    return ((pose >> surface(pose << Point), mv.bivector()),)


def sphere_family(weights: np.ndarray) -> Quadric:
    # The paper's diagonal sphere-model quadrics, expressed as dyad sums.
    spheres = Extensor.stack([mv.x, mv.y, mv.z, zero_sphere - infinity * 0.5])
    return (spheres * (spheres & Point) * weights).sum(axis=0)


def six_families():
    surface = sphere_family(np.array([-2, -1, 1, 2]))
    pose = (mv.xy * 0.25 + mv.yz * -0.15).exp()
    return ((pose >> surface(pose << Point), mv.bivector()),)


def inverted_hyperboloid():
    surface = (mv.z * (mv.z & Point) - mv.x * (mv.x & Point)
               - mv.y * (mv.y & Point) + infinity * (infinity & Point))
    surface = surface - (1 / 0.65**2 - 1) * mv.y * (mv.y & Point)
    shift = (-0.5 * ((mv.x * 1.7) ^ infinity)).exp()
    inversion = (shift >> (zero_sphere - infinity)).normalized()
    # Reverse the sign because this inversion centre lies outside the solid.
    return -(inversion >> surface(inversion << Point)), inversion


def hyperboloid():
    surface, _ = inverted_hyperboloid()
    return ((surface, mv.bivector()),)


def two_lobed():
    surface = sphere_family(np.array([-3, -0.25, 1, 1]))
    pose = (mv.xy * 0.12 + mv.yz * -0.12).exp()
    return ((pose >> surface(pose << Point), mv.bivector()),)


def vortex():
    surface = sphere_family(np.array([-3, -0.25, 1, 1]))
    circle = ((mv.z + mv.x * 0.2).normalized() ^ (zero_sphere - infinity * 0.5)).normalized()
    return ((surface, circle),)


def linked_vortex():
    surface, inversion = inverted_hyperboloid()
    # Invert a circle surrounding the original hyperboloid's waist together
    # with the body. The resulting vortex circle threads its opening transversely.
    radius = 2.5
    sphere = zero_sphere - infinity * (radius * radius / 2)
    circle = (inversion >> (mv.z ^ sphere)).normalized()
    # The inverted circle still lies in z = 0. Normalize its sphere to recover
    # its world-space radius, then give it a constant-thickness torus tube.
    sphere = inversion >> sphere
    sphere = sphere / -(sphere | infinity)
    radius_squared = sphere.squared()
    tube = 0.02
    sphere = sphere - infinity * (tube * tube / 2)
    ring = sphere * (sphere & Point) + radius_squared * (
        mv.z * (mv.z & Point) - tube * tube * infinity * (infinity & Point)
    )
    return ((surface, circle), (ring, mv.bivector()))


def linked_tori():
    # Matching centreline radii, with a small shift along the tilt axis.
    # Matching the shift to radius * sin(tilt) keeps the initial gap fairly even.
    radius, offset = 1.4, 0.45
    surface = torus(radius, 0.16)
    placement = (-0.5 * ((mv.x * offset) ^ infinity)).exp() * (
        mv.yz * (np.arcsin(offset / radius) / 2)
    ).exp()
    circle = (mv.z ^ (zero_sphere - infinity * (radius * radius / 2))).normalized()
    return ((placement >> surface(placement << Point), circle),
            (torus(radius, 0.02), mv.bivector()))


def pinched_vortex():
    surface = pinched_surface()
    placement = (-0.5 * ((mv.x * 1.8) ^ infinity)).exp() * (mv.xz * (np.pi / 4)).exp()
    radius = 1.4
    circle = (mv.z ^ (zero_sphere - infinity * (radius * radius / 2))).normalized()
    return ((placement >> surface(placement << Point), circle),
            (torus(radius, 0.02), mv.bivector()))


def open_hyperboloid():
    # Elliptical throat, opening indefinitely along z.
    surface = (mv.x * (mv.x & Point) + (mv.y * (mv.y & Point)) / 0.7**2
               - (mv.z * (mv.z & Point)) / 1.3**2 - infinity * (infinity & Point))
    return ((surface, mv.bivector()),)


def saddle():
    # The symmetric plane dyad supplies the linear height term: z = (x² - y²) / 2.
    surface = (mv.y * (mv.y & Point) - mv.x * (mv.x & Point)
               - mv.z * (infinity & Point) - infinity * (mv.z & Point))
    return ((surface, mv.bivector()),)


def cubic_cyclide():
    # The inversion sphere is centred exactly on the torus, sending that point to infinity.
    surface = torus(1.5, 0.5)
    shift = (-0.5 * ((mv.x * 2) ^ infinity)).exp()
    inversion = (shift >> (zero_sphere - infinity * 2)).normalized()
    return ((inversion >> surface(inversion << Point), mv.bivector()),)


def tunnel_wall() -> Quadric:
    # The spherical ++-- tunnel in the sphere model: y and the unit sphere
    # span its core, with a wide section at y = 1 and a narrow waist at y = 0.
    sphere = zero_sphere - infinity * 0.5
    return (-mv.x * (mv.x & Point) - 2.25 * mv.z * (mv.z & Point)
            + np.tan(np.radians(40))**2 * mv.y * (mv.y & Point)
            + np.tan(np.radians(20))**2 * sphere * (sphere & Point))


def tunnel():
    centers = mv(Direction, [
        [-0.22, 0.65, -0.14], [0.24, 0.5, 0.13], [-0.10, 0.28, 0.06],
        [0.10, 0.05, -0.025], [-0.065, -0.25, -0.065], [0.13, -0.65, 0.1],
    ])
    radii = np.array([0.09, 0.07, 0.045, 0.032, 0.045, 0.065])
    spheres = (-0.5 * (centers ^ infinity)).exp() >> (zero_sphere - infinity * (radii*radii/2))
    bodies = spheres * (infinity & Point) + infinity * (spheres & Point)
    return ((tunnel_wall(), mv.bivector()),) + tuple((body, mv.bivector()) for body in bodies)


def hyperbolic_tunnel():
    # Positive inside the throat, so the polarity gives inward wall normals.
    wall = (infinity * (infinity & Point) + mv.y * (mv.y & Point) / 2.2**2
            - mv.x * (mv.x & Point) / 0.85**2 - mv.z * (mv.z & Point) / 0.6**2)
    return ((wall, mv.bivector()),)


def bent_tunnel():
    # Both circles pass through the eye: their vortices bend the channel while
    # leaving the camera in place. Their different planes break axial symmetry.
    eye = mv(Direction, [0.06, 1, 0.03])
    center = eye - mv.y * 0.8
    side_circle = (-0.5 * (center ^ infinity)).exp() >> (
        (mv.x ^ (zero_sphere - infinity * (0.8**2 / 2))) / 0.8
    )
    center = eye + mv.x * 0.65
    upper_circle = (-0.5 * (center ^ infinity)).exp() >> (
        (mv.z ^ (zero_sphere - infinity * (0.65**2 / 2))) / 0.65
    )
    warp = (upper_circle * -0.16).exp() * (side_circle * 0.32).exp()
    wall = warp >> tunnel_wall()(warp << Point)

    # Narrow the tubes on opposite sides before bending the rings with the passage.
    centers = mv(Direction, [[-0.04, 0.5, 0.015], [0.015, -0.05, 0], [-0.02, -0.65, 0]])
    radii = np.array([0.16, 0.12, 0.17])
    rings = torus(radii, radii * 0.25) + (0.035 * radii**2) * mv.y * (mv.y & Point)
    placement = (warp * (-0.5 * (centers ^ infinity)).exp()
                 * (mv.xz * (0.12 + np.arange(3) * 0.15)).exp()
                 * (mv.yz * (np.pi / 4 + 0.2)).exp())
    rings = placement >> rings(placement << Point)
    return ((wall, mv.bivector()),) + tuple((ring, mv.bivector()) for ring in rings)


SCENES = {
    "cyclide": (cyclide, [0.2, -8.5, 5.8], [-0.8, 0, 0], 43),
    "peanut": (peanut, [0.2, -7, 4.5], [0, 0, 0], 40),
    "elliptic_ring": (elliptic_ring, [0.2, -7, 4.5], [0, 0, 0], 40),
    "split_ring": (split_ring, [0.2, -7, 4.5], [0, 0, 0], 40),
    "pinched": (pinched, [0.2, -7, 4.5], [0, 0, 0], 40),
    "six_families": (six_families, [0.6, -9, 6.5], [0, 0, 0], 40),
    "hyperboloid": (hyperboloid, [10, -2, 1.8], [0.6, 0, 0], 40),
    "two_lobed": (two_lobed, [0.5, -12, 8], [0, 0, 0], 40),
    "vortex": (vortex, [0.6, -15, 10.5], [0, 0, 0], 44),
    "linked_vortex": (linked_vortex, [2.8, -9, 4.8], [1, 0, 0], 44),
    "linked_tori": (linked_tori, [0.5, -7, 4.8], [0, 0, 0], 40),
    "pinched_vortex": (pinched_vortex, [0.5, -10, 7], [0.5, 0, 0], 44),
    "open_hyperboloid": (open_hyperboloid, [5, -8, 3.5], [0, 0, 0], 48),
    "saddle": (saddle, [7, -3, 5], [0, 0, 0], 60),
    "cubic_cyclide": (cubic_cyclide, [0.5, -8, 3], [1, 0, 0], 50),
    "tunnel": (tunnel, [0.06, 1, 0.03], [0, -0.4, 0], 105),
    "hyperbolic_tunnel": (hyperbolic_tunnel, [0.15, -3.5, 0.1], [0, 1.5, 0], 100),
    "bent_tunnel": (bent_tunnel, [0.06, 1, 0.03], [0.4, -0.4, 0.08], 95),
}


@dataclass
class Scene:
    name: str
    surfaces: Quadric
    motors: Extensor
    position: Direction
    target: Direction
    fov: float
    colors: np.ndarray
    main_light_source: Sphere
    fill_light_source: Sphere
    shape: tuple[int, int] = (600, 800)
    supersample: int = 2


def directional_lights(position: Direction) -> tuple[Sphere, Sphere]:
    return mv(Direction, [-3, -4, 7]).normalized(), mv(Direction, [4, 1, 2]).normalized()


def tunnel_lights(position: Direction) -> tuple[Sphere, Sphere]:
    # Small null spheres are point sources; place the main light halfway to
    # the upper-right wall, keeping both sources attached to the camera.
    main = (-0.5 * ((position + mv.x*0.23 + mv.z*0.20) ^ infinity)).exp() >> zero_sphere
    fill = (-0.5 * ((position - mv.x*0.20 - mv.z*0.08) ^ infinity)).exp() >> zero_sphere
    return main, fill


LIGHTING = {"tunnel": tunnel_lights, "hyperbolic_tunnel": tunnel_lights, "bent_tunnel": tunnel_lights}


def scenario(name: str, frame_count: int, shape: tuple[int, int] = (600, 800), supersample: int = 2) -> Scene:
    build, position, target, degrees = SCENES[name]
    parts = build()
    phase = np.arange(frame_count) * (2 * np.pi / frame_count)
    motors = Extensor.stack([(generator * (phase / 2)).exp() for _, generator in parts], axis=1)
    position = mv(Direction, position)
    main_light, fill_light = LIGHTING.get(name, directional_lights)(position)
    colors = np.resize(np.array([[0.055, 0.42, 0.39], [0.65, 0.38, 0.08],
                                 [0.35, 0.13, 0.40], [0.12, 0.35, 0.60]]), (len(parts), 3))
    return Scene(
        name, Extensor.stack([surface for surface, _ in parts]), motors,
        position, mv(Direction, target), np.radians(degrees),
        colors, main_light, fill_light,
        shape, supersample,
    )


def main(
    name: str = "cyclide", frame_count: int = 1,
    shape: tuple[int, int] = (600, 800), supersample: int = 2,
    output_dir: Path = PLOT_DIR, scale: float = 1.0, duration_ms: int = 100,
) -> str:
    from examples.sketches.cga_quadric import render
    from examples.sketches.cga_quadric_plumbing import export

    scene = scenario(name, frame_count, shape, supersample)
    return export(render(scene), name, output_dir, scale, duration_ms)
