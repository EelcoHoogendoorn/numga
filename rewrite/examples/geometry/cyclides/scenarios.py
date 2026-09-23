"""Scenes of Dupin cyclides on the 3-sphere: tori, a torus rolling around a circle, lopsided Dupin cyclides, and spindle cyclides.

Run from rewrite/:
    PYTHONPATH=src:. python -m examples.geometry.cyclides.scenarios
"""

from __future__ import annotations

import numpy as np

from collections.abc import Iterator

from numga import stack
from examples.geometry.cyclides.core import Motor, Point, Scalar, Sphere, cone, cylinder, dilation, mv, sensor, trace

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
    circle = ((place * (mv.yw * 0.2).exp()) >> (mv.z ^ mv.w)).restrict[2]      # the meet of two spheres
    circle = circle / (-(circle * circle).select[0]).square_root()             # circle * circle == -1
    flow = (circle * (np.linspace(0.0, 2 * np.pi, frames, endpoint=False) / 2)).exp()
    for surface in flow >> torus(flow << Point):
        yield trace(surface.reshape(1), PIXELS)


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

    # The eye aims at the midpoint of the two dilated vertices, z and -z carried by the dilation.
    vertices = dilations[:, None] >> stack((mv.z + mv.e, -mv.z + mv.e))           # null vectors
    unit_e = vertices / -(vertices | mv.e)                                    # one unit of e each
    midpoints = (unit_e[:, 0] + unit_e[:, 1] - 2 * mv.e).normalized()         # points of S³ as unit vectors

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
    facing, visible = trace(views >> surfaces(views << Point), PIXELS)
    return facing.abs(), visible


if __name__ == "__main__":
    from examples.animation import save_animation, save_figure
    from examples.geometry.cyclides import render

    save_figure(render.draw_facing(*tori(), SHAPE), "cyclides_tori")
    save_animation(render.facing_frames(vortex(48), SHAPE), "cyclides_vortex", 60)
    save_figure(render.draw_facing(*dupin(), SHAPE), "cyclides_dupin")
    save_figure(render.draw_facing(*spindles(), SHAPE),
                "cyclides_spindles")
