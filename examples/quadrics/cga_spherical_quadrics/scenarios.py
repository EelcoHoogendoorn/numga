"""Spherical quadric shapes on S², and the conformal vortex that carries them."""

from __future__ import annotations

import numpy as np

from numga import Extensor
from examples.quadrics.cga_spherical_quadrics.core import (
    Quadric, flow, make_bernoulli_lemniscate, make_circle_intersection_vortex, make_conical_donut,
    make_eccentric_cyclide, make_pinched_horn, make_spherical_cassini, make_spherical_clover,
    make_spherical_crescent, make_spherical_hourglass, make_spherical_parabola, make_spherical_spindle,
    moved, mv, point,
)


def shapes() -> dict[str, tuple[Quadric, list[str]]]:
    """Every shape, tilted into view, with its colour; the trio combines three of them."""
    single = {
        # Donut (solid ring and central hole) that narrows and meets in a sharp conical apex.
        "conical_donut": (moved(make_conical_donut(0.55, 0.22), (mv.xz * -0.20).exp()), "#38bdf8"),
        # Pinched-waist Cassini oval.
        "peanut": (moved(make_spherical_cassini(np.radians(25.0), np.radians(25.0), 0.012), (mv.yz * 0.30).exp()), "#f59e0b"),
        # Two disconnected droplet beads.
        "islands": (moved(make_spherical_cassini(np.radians(25.0), np.radians(25.0), 0.006), (mv.yz * 0.30).exp()), "#10b981"),
        # Figure-8 with two symmetric lobes crossing at a node.
        "lemniscate": (make_bernoulli_lemniscate(0.38), "#a855f7"),
        # Eccentric sickle with two sharp cusps.
        "crescent": (moved(make_spherical_crescent(np.radians(45.0), np.radians(24.0), np.radians(16.0)), (mv.yz * 0.25).exp()), "#fb7185"),
        # Eccentric Dupin cyclide with unequal tube width.
        "cyclide": (moved(make_eccentric_cyclide(np.radians(34.0), np.radians(13.0), 0.65), (mv.yz * 0.30).exp()), "#fbbf24"),
        # Pinched horn cyclide with a singular self-touching cusp.
        "pinched": (moved(make_pinched_horn(np.radians(65.0)), (mv.yz * 0.35).exp()), "#f43f5e"),
        # Spindle with two sharp opposite conical poles.
        "spindle": (moved(make_spherical_spindle(1.8, 0.8, 0.35), (mv.yz * 0.30).exp()), "#06b6d4"),
        # Asymmetric Cassini droplet tapering to a fine tail.
        "teardrop": (moved(make_spherical_cassini(np.radians(10.0), np.radians(35.0), 0.010), (mv.yz * 0.30).exp()), "#ec4899"),
        # Triadic 3-lobed deltoid.
        "clover": (moved(make_spherical_clover(np.radians(32.0), np.radians(28.0), 0.35), (mv.yz * 0.20).exp()), "#4ade80"),
        # Vertical hourglass: two symmetric bells connected by a narrow waist.
        "hourglass": (make_spherical_hourglass(np.radians(24.0), 0.010), "#f59e0b"),
        # Parabolic bow.
        "parabola": (moved(make_spherical_parabola(1.0, 0.5, 0.20), (mv.yz * 0.25).exp()), "#f97316"),
    }
    table = {name: (Extensor.stack([quadric]), [color]) for name, (quadric, color) in single.items()}
    trio = ("conical_donut", "lemniscate", "hourglass")
    table["trio"] = (Extensor.stack([single[name][0] for name in trio]), [single[name][1] for name in trio])
    return table


def vortex(names: list[str], frames: int) -> tuple[Quadric, list[str]]:
    """The chosen shapes carried once around by the exponential of two circles' intersection."""
    table = shapes()
    quadrics = Extensor.concatenate([table[name][0] for name in names])
    colors = [color for name in names for color in table[name][1]]
    generator = make_circle_intersection_vortex(
        (mv.zx * (-0.35 / 2)).exp() >> mv.z, np.radians(48.0),
        (mv.zy * (-0.40 / 2)).exp() >> mv.z, np.radians(52.0),
    )
    world = flow(quadrics, generator, np.linspace(0.0, 2.0 * np.pi, frames, endpoint=False))

    # --- checks ---------------------------------------------------------------------------
    # The flow is conformal, so points of S² stay null; and a full turn brings every quadric back.
    probes = point(np.array([[0.6, 0.0, 0.8], [0.0, -0.28, 0.96], [0.36, 0.48, 0.8]]))
    carried = (generator * 0.4).exp() >> probes
    np.testing.assert_allclose((carried | carried).to_array(), 0.0, atol=1e-12)
    returned = flow(quadrics, generator, np.array([2.0 * np.pi]))[0]
    np.testing.assert_allclose((probes[:, None] & returned(probes[:, None])).to_array(),
                               (probes[:, None] & quadrics(probes[:, None])).to_array(), atol=1e-6)
    return world, colors


if __name__ == "__main__":
    import argparse
    from examples.animation import save_animation
    from examples.quadrics.cga_spherical_quadrics import render

    parser = argparse.ArgumentParser(description="Spherical quadric vortex in Cl(3, 1)")
    parser.add_argument("--shape", type=str, default="trio", help="comma-separated names: " + ", ".join(shapes()))
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--resolution", type=int, default=300)
    parser.add_argument("--supersample", type=int, default=2)
    args = parser.parse_args()

    names = [name.strip() for name in args.shape.split(",")]
    world, colors = vortex(names, args.frames)
    save_animation(render.vortex_frames(world, colors, args.resolution, args.supersample), f"cga_{'_'.join(names)}_vortex", 33)
