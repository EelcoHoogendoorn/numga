"""Spherical quadric shapes on S², and the conformal vortex that carries them."""

from __future__ import annotations

import numpy as np

from numga import concatenate, stack
from examples.quadrics.cga_spherical_quadrics.core import (
    Point, Quadric, flow, make_bernoulli_lemniscate, make_conical_donut, make_eccentric_cyclide, make_pinched_horn,
    make_spherical_cassini, make_spherical_clover, make_spherical_crescent, make_spherical_hourglass,
    make_spherical_parabola, make_spherical_spindle, mv, point,
)

# The two circles whose intersection generates the flow: each the plane z tilted toward x or y, and its radius.
FIRST_TILT, FIRST_RADIUS = 0.35, np.radians(48.0)
SECOND_TILT, SECOND_RADIUS = 0.40, np.radians(52.0)
# The shapes carried, by name in `shapes`; the frames of one turn, the pixels across, and a frame's duration in
# milliseconds.
shape_names = ["trio"]
n_animation_frames = 50
resolution = 300
frame_duration = 40


# The two circles, each the vector of its plane with the cosine of its radius on w, meet in a 2-blade: its
# exponential is the conformal flow.
first = ((mv.zx * (-FIRST_TILT / 2)).exp() >> mv.z) + mv.w * np.cos(FIRST_RADIUS)        # [] Plane+
second = ((mv.zy * (-SECOND_TILT / 2)).exp() >> mv.z) + mv.w * np.cos(SECOND_RADIUS)     # [] Plane+
generator = (first ^ second).normalized()                                                # [] Bivector+


def shapes() -> dict[str, tuple[Quadric, list[str]]]:
    """Every shape, tilted into view by a rotor, with its colour; the trio combines three of them."""
    single = {
        # Donut (solid ring and central hole) that narrows and meets in a sharp conical apex.
        "conical_donut": (make_conical_donut(0.55, 0.22), (mv.xz * -0.20).exp(), "#38bdf8"),
        # Pinched-waist Cassini oval.
        "peanut": (make_spherical_cassini(np.radians(25.0), np.radians(25.0), 0.012), (mv.yz * 0.30).exp(), "#f59e0b"),
        # Two disconnected droplet beads.
        "islands": (make_spherical_cassini(np.radians(25.0), np.radians(25.0), 0.006), (mv.yz * 0.30).exp(), "#10b981"),
        # Figure-8 with two symmetric lobes crossing at a node.
        "lemniscate": (make_bernoulli_lemniscate(0.38), (mv.yz * 0.0).exp(), "#a855f7"),
        # Eccentric sickle with two sharp cusps.
        "crescent": (make_spherical_crescent(np.radians(45.0), np.radians(24.0), np.radians(16.0)), (mv.yz * 0.25).exp(), "#fb7185"),
        # Eccentric Dupin cyclide with unequal tube width.
        "cyclide": (make_eccentric_cyclide(np.radians(34.0), np.radians(13.0), 0.65), (mv.yz * 0.30).exp(), "#fbbf24"),
        # Pinched horn cyclide with a singular self-touching cusp.
        "pinched": (make_pinched_horn(np.radians(65.0)), (mv.yz * 0.35).exp(), "#f43f5e"),
        # Spindle with two sharp opposite conical poles.
        "spindle": (make_spherical_spindle(1.8, 0.8, 0.35), (mv.yz * 0.30).exp(), "#06b6d4"),
        # Asymmetric Cassini droplet tapering to a fine tail.
        "teardrop": (make_spherical_cassini(np.radians(10.0), np.radians(35.0), 0.010), (mv.yz * 0.30).exp(), "#ec4899"),
        # Triadic 3-lobed deltoid.
        "clover": (make_spherical_clover(np.radians(32.0), np.radians(28.0), 0.35), (mv.yz * 0.20).exp(), "#4ade80"),
        # Vertical hourglass: two symmetric bells connected by a narrow waist.
        "hourglass": (make_spherical_hourglass(np.radians(24.0), 0.010), (mv.yz * 0.0).exp(), "#f59e0b"),
        # Parabolic bow.
        "parabola": (make_spherical_parabola(1.0, 0.5, 0.20), (mv.yz * 0.25).exp(), "#f97316"),
    }
    # Each shape carried into view by its tilt: the point pulled back, the plane pushed forward.
    tilted = {name: (tilt >> quadric(tilt << Point), color) for name, (quadric, tilt, color) in single.items()}
    table = {name: (stack([quadric]), [color]) for name, (quadric, color) in tilted.items()}
    trio = ("conical_donut", "lemniscate", "hourglass")
    table["trio"] = (stack([tilted[name][0] for name in trio]), [tilted[name][1] for name in trio])
    return table


def vortex(names: list[str], n_frames: int) -> tuple[Quadric, list[str]]:
    """The chosen shapes carried once around by the exponential of two circles' intersection."""
    table = shapes()
    quadrics = concatenate([table[name][0] for name in names])
    colors = [color for name in names for color in table[name][1]]
    world = flow(quadrics, generator, np.linspace(0.0, 2.0 * np.pi, n_frames, endpoint=False))
    return world, colors


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation
    from examples.quadrics.cga_spherical_quadrics import render

    world, colors = vortex(shape_names, n_animation_frames)
    save_animation(render.vortex_frames(world, colors, resolution), f"cga_{'_'.join(shape_names)}_vortex", frame_duration)

    # checks
    # The flow is conformal, so points of S² stay null; and a full turn brings every quadric back.
    quadrics = concatenate([shapes()[name][0] for name in shape_names])
    probes = point(np.array([[0.6, 0.0, 0.8], [0.0, -0.28, 0.96], [0.36, 0.48, 0.8]]))
    carried = (generator * 0.4).exp() >> probes
    np.testing.assert_allclose((carried | carried).to_array(), 0.0, atol=1e-12)
    returned = flow(quadrics, generator, np.array([2.0 * np.pi]))[0]
    np.testing.assert_allclose((returned(probes[:, None]) & probes[:, None]).to_array(),
                               (quadrics(probes[:, None]) & probes[:, None]).to_array(), atol=1e-6)


if __name__ == "__main__":
    main()
