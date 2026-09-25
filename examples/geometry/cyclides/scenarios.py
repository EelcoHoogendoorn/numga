"""Scenes of Dupin cyclides on the 3-sphere: tori, a torus rolling around a circle, lopsided Dupin cyclides, and spindle cyclides.

Run from the repository root:
    python -m examples.geometry.cyclides.scenarios
"""

from __future__ import annotations

import numpy as np

from collections.abc import Iterator

from numga import Extensor, stack
from examples.geometry.cyclides.core import (
    Direction, Motor, Point, Quadric, Scalar, Sphere, chart_infinity, chart_origin, cone, cylinder, dilation, mv, sensor,
    trace,
)

SHAPE = (240, 320)
PIXELS = sensor(SHAPE, np.radians(90))


def placement(toward: Sphere, ahead: np.ndarray, tilt: np.ndarray) -> Motor:
    """Turn the point `toward` onto the point `ahead` rad in front of the eye, after a tilt that keeps it fixed."""
    target = mv.w * np.cos(ahead) - mv.x * np.sin(ahead)
    return (1 + target * toward).normalized() * (mv.yw * (tilt / 2)).exp()


def orientation(angles: list[float]) -> Motor:
    """A rotation of R⁴ by the given angles in the planes xy, xz, xw, yz, yw and zw, in turn."""
    rotor = mv.rotor()
    for plane, angle in zip((mv.xy, mv.xz, mv.xw, mv.yz, mv.yw, mv.zw), angles):
        rotor = rotor * (plane * angle).exp()
    return rotor


def view(centre: Sphere, angles: list[float], ahead: float, yaw: float, pitch: float) -> Motor:
    """Turn the shape, carry its centre to `ahead` rad in front of the eye, then turn the eye onto it."""
    turned = orientation(angles)
    target = mv.w * np.cos(ahead) - mv.x * np.sin(ahead)
    carry = (1 + target * (turned >> centre)).normalized()
    look = (mv.xy * (yaw / 2)).exp() * (mv.xz * (pitch / 2)).exp()
    return look * carry * turned


def tori() -> tuple[Scalar, np.ndarray]:
    """Cylinders with tubes of 0.15, 0.5 and 0.9 rad, dilated towards z, a quarter turn from their whole
    core: the core shrinks evenly into a circle, and each tube becomes a torus. Seen from z the
    dilation is a uniform scaling, so it only sizes the torus; the tube sets its shape."""
    place = placement(mv.z, 1.1, 0.8) * dilation(mv.z, np.array([1.55, 2.0, 2.4]))
    return trace(place >> cylinder(np.array([0.15, 0.5, 0.9]))(place << Point), PIXELS)


def vortex(frames: int) -> Iterator[tuple[Scalar, np.ndarray]]:
    """The middle torus carried once around a circle: its own core circle, tilted 0.4 rad in the yw
    plane. Around the core itself the flow would only spin the tube in place; tilted, it rolls and
    deforms the torus. Yields each frame's trace as it is consumed."""
    place = placement(mv.z, 1.1, 0.8) * dilation(mv.z, 2.0)
    torus = place >> cylinder(0.5)(place << Point)
    # The meet of two great spheres, carried by unit versors: a unit circle, circle * circle == -1.
    circle = (place * (mv.yw * 0.2).exp()) >> (mv.z ^ mv.w)
    flow = (circle * (np.linspace(0.0, 2 * np.pi, frames, endpoint=False) / 2)).exp()
    for surface in flow >> torus(flow << Point):
        yield trace(surface.reshape(1), PIXELS)


# --- the flat tracer's scenes -----------------------------------------------------------
# The shapes of the flat conformal tracer, built in the chart about the eye, and its table of scenes: each a list
# of (surface, vortex circle) parts, a camera position and target, and a field of view in degrees.
def torus(radius: float, tube: float) -> Quadric:
    sphere = chart_origin - chart_infinity * ((radius * radius + tube * tube) / 2)
    return sphere * (sphere & Point) + radius * radius * (
        mv.z * (mv.z & Point) - tube * tube * chart_infinity * (chart_infinity & Point)
    )


def cyclide():
    surface = torus(1.4, 0.5)
    # An off-centre sphere inversion gives the torus unequal tube widths.
    shift = (-0.5 * ((mv.x * 3.2) ^ chart_infinity)).exp()
    inversion = (shift >> (chart_origin - chart_infinity * 4.5)).normalized()
    pose = (mv.xy * 0.12).exp() * (mv.yz * -0.22).exp()
    placement = pose * inversion
    return ((placement >> surface(placement << Point), mv.bivector()),)


def peanut():
    a, b = 1.25, 1.31
    sphere = chart_origin - chart_infinity * (a * a / 2)
    surface = sphere * (sphere & Point) + a * a * (
        mv.y * (mv.y & Point) + mv.z * (mv.z & Point)
    ) - (b**4 / 4) * chart_infinity * (chart_infinity & Point)
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
               - mv.y * (mv.y & Point) + chart_infinity * (chart_infinity & Point))
    inversion = (chart_origin - chart_infinity).normalized()
    return inversion >> surface(inversion << Point)


def pinched():
    surface = pinched_surface()
    pose = (mv.yz * -0.18).exp()
    return ((pose >> surface(pose << Point), mv.bivector()),)


def sphere_family(weights: np.ndarray) -> Quadric:
    # The paper's diagonal sphere-model quadrics, expressed as dyad sums.
    spheres = Extensor.stack([mv.x, mv.y, mv.z, chart_origin - chart_infinity * 0.5])
    return (spheres * (spheres & Point) * weights).sum(axis=0)


def six_families():
    surface = sphere_family(np.array([-2, -1, 1, 2]))
    pose = (mv.xy * 0.25 + mv.yz * -0.15).exp()
    return ((pose >> surface(pose << Point), mv.bivector()),)


def inverted_hyperboloid():
    surface = (mv.z * (mv.z & Point) - mv.x * (mv.x & Point)
               - mv.y * (mv.y & Point) + chart_infinity * (chart_infinity & Point))
    surface = surface - (1 / 0.65**2 - 1) * mv.y * (mv.y & Point)
    shift = (-0.5 * ((mv.x * 1.7) ^ chart_infinity)).exp()
    inversion = (shift >> (chart_origin - chart_infinity)).normalized()
    # Reverse the sign because this inversion centre lies outside the solid.
    return -(inversion >> surface(inversion << Point)), inversion


def hyperboloid():
    surface, _ = inverted_hyperboloid()
    return ((surface, mv.bivector()),)


def two_lobed():
    surface = sphere_family(np.array([-3, -0.25, 1, 1]))
    pose = (mv.xy * 0.12 + mv.yz * -0.12).exp()
    return ((pose >> surface(pose << Point), mv.bivector()),)


def linked_vortex():
    surface, inversion = inverted_hyperboloid()
    # Invert a circle surrounding the uninverted hyperboloid's waist together
    # with the body. The resulting vortex circle threads its opening transversely.
    radius = 2.5
    sphere = chart_origin - chart_infinity * (radius * radius / 2)
    circle = (inversion >> (mv.z ^ sphere)).normalized()
    # The inverted circle still lies in the plane z == 0. Normalize its sphere to recover
    # its world-space radius, then give it a constant-thickness torus tube.
    sphere = inversion >> sphere
    sphere = sphere / -(sphere | chart_infinity)
    radius_squared = sphere.squared()
    tube = 0.02
    sphere = sphere - chart_infinity * (tube * tube / 2)
    ring = sphere * (sphere & Point) + radius_squared * (
        mv.z * (mv.z & Point) - tube * tube * chart_infinity * (chart_infinity & Point)
    )
    return ((surface, circle), (ring, mv.bivector()))


def linked_tori():
    # Matching centreline radii, with a small shift along the tilt axis.
    # Matching the shift to radius * np.sin(tilt) keeps the initial gap fairly even.
    radius, offset = 1.4, 0.45
    surface = torus(radius, 0.16)
    placement = (-0.5 * ((mv.x * offset) ^ chart_infinity)).exp() * (
        mv.yz * (np.arcsin(offset / radius) / 2)
    ).exp()
    circle = (mv.z ^ (chart_origin - chart_infinity * (radius * radius / 2))).normalized()
    return ((placement >> surface(placement << Point), circle),
            (torus(radius, 0.02), mv.bivector()))


def pinched_vortex():
    surface = pinched_surface()
    placement = (-0.5 * ((mv.x * 1.8) ^ chart_infinity)).exp() * (mv.xz * (np.pi / 4)).exp()
    radius = 1.4
    circle = (mv.z ^ (chart_origin - chart_infinity * (radius * radius / 2))).normalized()
    return ((placement >> surface(placement << Point), circle),
            (torus(radius, 0.02), mv.bivector()))


FLAT_SCENES = {
    "cyclide": (cyclide, [0.2, -8.5, 5.8], [-0.8, 0, 0], 43),
    "peanut": (peanut, [0.2, -7, 4.5], [0, 0, 0], 40),
    "elliptic_ring": (elliptic_ring, [0.2, -7, 4.5], [0, 0, 0], 40),
    "split_ring": (split_ring, [0.2, -7, 4.5], [0, 0, 0], 40),
    "pinched": (pinched, [0.2, -7, 4.5], [0, 0, 0], 40),
    "six_families": (six_families, [0.6, -9, 6.5], [0, 0, 0], 40),
    "hyperboloid": (hyperboloid, [10, -2, 1.8], [0.6, 0, 0], 40),
    "two_lobed": (two_lobed, [0.5, -12, 8], [0, 0, 0], 40),
    "linked_vortex": (linked_vortex, [2.8, -9, 4.8], [1, 0, 0], 44),
    "linked_tori": (linked_tori, [0.5, -7, 4.8], [0, 0, 0], 40),
    "pinched_vortex": (pinched_vortex, [0.5, -10, 7], [0.5, 0, 0], 44),
}


def flat_camera(position: Direction, target: Direction) -> Motor:
    """The motor that brings the flat tracer's camera to the eye: its pose carries a camera at the origin, looking
    along -x with z up, to `position` looking at `target`, and the eye of S³ sits at the chart's origin looking
    along -x. Great circles through the eye are the chart's straight lines through the origin, so the view is the
    flat tracer's."""
    forward = (target - position).normalized()
    right = ((forward ^ mv.z) * mv.xyz.inverse()).normalized()
    aim = (1 - forward * mv.x).normalized()
    roll = (1 + right * (aim >> mv.y)).normalized()
    return ((-0.5 * (position ^ chart_infinity)).exp() * roll * aim).inverse()


def flat_scene(name: str, frames: int) -> Iterator[tuple[Scalar, np.ndarray]]:
    """One of the flat tracer's scenes on S³: every part turned around its own vortex circle over the frames, all
    traced from the eye. Yields each frame's trace of the parts; both sides are lit."""
    build, position, target, degrees = FLAT_SCENES[name]
    parts = build()
    surfaces = stack([surface for surface, _ in parts])
    camera = flat_camera(mv(Direction, position), mv(Direction, target))
    pixels = sensor(SHAPE, np.radians(degrees))
    for phase in np.linspace(0.0, 2 * np.pi, frames, endpoint=False):
        motion = camera * stack([(circle * (phase / 2)).exp() for _, circle in parts])
        facing, angle = trace(motion >> surfaces(motion << Point), pixels)
        yield facing.abs(), angle


def dupin() -> tuple[Scalar, np.ndarray]:
    """Dilations aimed off z, leaning towards the core point x: the tube is compressed on the side of
    the aim and swells on the other, the lopsided Dupin cyclides."""
    leans = np.array([0.7, 0.8, 0.5])
    aims = mv.z * np.cos(leans) + mv.x * np.sin(leans)
    dilations = dilation(aims, np.array([1.5, 1.2, 1.8]))
    bent = dilations >> cylinder(np.array([0.25, 0.3, 0.2]))(dilations << Point)
    views = placement(aims, np.array([1.2, 1.3, 1.1]), np.array([-0.7, 0.6, 1.0]))
    return trace(views >> bent(views << Point), PIXELS)


def spindles() -> tuple[Scalar, np.ndarray]:
    """The cone with half-opening 0.35 dilated strongly towards its axis point, straight and leaning
    towards y: its two vertices close into a spindle cyclide, and the leaning dilation bends the inner
    sheet into a banana. The eye sees both sides of the sheets, so facing is the unsigned cosine."""
    leans = np.array([0.0, 0.7])
    dilations = dilation(mv.x * np.cos(leans) + mv.y * np.sin(leans), 3.0)
    bent = dilations >> cone(0.35)(dilations << Point)

    # The eye aims at the midpoint of the two dilated vertices, z and -z carried by the dilation. The
    # vertices are null vectors; scaled to one unit of e each, their sum less its e part, normalized, is
    # the midpoint, a point of S³ as a unit vector.
    vertices = dilations[:, None] >> stack((mv.z + mv.e, -mv.z + mv.e))           # [spindles, 2] Sphere
    unit_e = vertices / -(vertices | mv.e)                                    # [spindles, 2] Sphere
    midpoints = (unit_e[:, 0] + unit_e[:, 1] - 2 * mv.e).normalized()         # [spindles] Sphere

    # Which spindle, its orientation in the six planes, the distance, and the eye's yaw and pitch.
    shots = [
        (0, [-0.186, -0.692, 2.658, 0.181, -0.282, -0.225], 1.10, -0.021, 0.061),
        (0, [1.124, 0.598, 0.155, 0.889, -0.164, -0.741], 1.05, -0.051, -0.062),
        (0, [0.464, 0.073, 0.536, -2.263, 0.817, -0.768], 1.05, -0.001, 0.060),
        (1, [-0.257, 0.18, 0.46, -0.999, -1.384, -0.004], 0.85, 0.137, -0.377),
        (1, [-0.511, -0.64, -0.64, 1.096, -1.168, -0.477], 0.85, -0.297, -0.358),
    ]
    views = stack([view(midpoints[i], angles, ahead, yaw, pitch) for i, angles, ahead, yaw, pitch in shots])
    surfaces = stack([bent[i] for i, *_ in shots])
    facing, angle = trace(views >> surfaces(views << Point), PIXELS)
    return facing.abs(), angle


if __name__ == "__main__":
    from examples.animation import save_animation, save_figure
    from examples.geometry.cyclides import render

    save_figure(render.draw_facing(*tori(), SHAPE), "cyclides_tori")
    save_animation(render.facing_frames(vortex(48), SHAPE), "cyclides_vortex", 60)
    for name, (build, *_) in FLAT_SCENES.items():
        animated = any(circle.kernel.any() for _, circle in build())
        save_animation(render.scene_frames(flat_scene(name, 48 if animated else 1), render.PALETTE, SHAPE), f"cyclides_{name}", 60)
    save_figure(render.draw_facing(*dupin(), SHAPE), "cyclides_dupin")
    save_figure(render.draw_facing(*spindles(), SHAPE),
                "cyclides_spindles")
