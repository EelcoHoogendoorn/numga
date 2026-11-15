"""Named spherical quadric presets and the intersecting-circle conformal vortex scenario."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import concatenate, stack
from examples.quadrics.cga_spherical_quadrics.core import (
    Point,
    Quadric,
    bernoulli_lemniscate,
    conformal_flow,
    conical_donut,
    eccentric_cyclide,
    mv,
    pinched_horn,
    sphere_points,
    spherical_cassini,
    spherical_clover,
    spherical_crescent,
    spherical_hourglass,
    spherical_parabola,
    spherical_spindle,
)

# Tilts and angular radii of the two off-axis circles whose intersection 2-blade drives the conformal vortex.
first_tilt, first_radius = 0.35, np.radians(48)
second_tilt, second_radius = 0.40, np.radians(52)
# Preset names rendered in `main`, frame count for one period, image resolution in pixels, and frame duration in ms.
shape_names = ["trio"]
n_frames = 50
resolution = 300
frame_duration_ms = 40

# Each circle is a tilted plane shifted by the cosine of its angular radius along w; their wedge generates the flow.
first_turn = (mv.zx * (-first_tilt / 2)).exp()                                            # [] Rotor+
second_turn = (mv.zy * (-second_tilt / 2)).exp()                                          # [] Rotor+
first_circle = (first_turn >> mv.z) + np.cos(first_radius) * mv.w                         # [] Plane+
second_circle = (second_turn >> mv.z) + np.cos(second_radius) * mv.w                      # [] Plane+
generator = (first_circle ^ second_circle).normalized()                                   # [] Bivector+


def shape_catalogue() -> dict[str, tuple[Quadric, list[str]]]:
    """Twelve named spherical quadrics tilted into view with their display colours, plus the three-shape `trio` stack
    (`[n_quadrics] Plane+ <- Point-`)."""
    single = {
        # Ring narrowing to a conical apex at the north pole.
        "conical_donut": (conical_donut(0.55, 0.22), (mv.xz * -0.20).exp(), "#38bdf8"),
        # Pinched-waist Cassini oval.
        "peanut": (spherical_cassini(np.radians(25), np.radians(25), 0.012), (mv.yz * 0.30).exp(), "#f59e0b"),
        # Disconnected twin Cassini islands.
        "islands": (spherical_cassini(np.radians(25), np.radians(25), 0.006), (mv.yz * 0.30).exp(), "#10b981"),
        # Figure-eight with two symmetric lobes crossing at a node.
        "lemniscate": (bernoulli_lemniscate(0.38), (mv.yz * 0.0).exp(), "#a855f7"),
        # Eccentric sickle with two cusps.
        "crescent": (spherical_crescent(np.radians(45), np.radians(24), np.radians(16)), (mv.yz * 0.25).exp(), "#fb7185"),
        # Eccentric Dupin cyclide with unequal tube width.
        "cyclide": (eccentric_cyclide(np.radians(34), np.radians(13), 0.65), (mv.yz * 0.30).exp(), "#fbbf24"),
        # Pinched horn cyclide with a self-touching cusp.
        "pinched": (pinched_horn(np.radians(65)), (mv.yz * 0.35).exp(), "#f43f5e"),
        # Spindle with two opposite conical poles.
        "spindle": (spherical_spindle(1.8, 0.8, 0.35), (mv.yz * 0.30).exp(), "#06b6d4"),
        # Asymmetric Cassini droplet tapering to a tail.
        "teardrop": (spherical_cassini(np.radians(10), np.radians(35), 0.010), (mv.yz * 0.30).exp(), "#ec4899"),
        # Three-lobed rounded deltoid.
        "clover": (spherical_clover(np.radians(32), np.radians(28), 0.35), (mv.yz * 0.20).exp(), "#4ade80"),
        # Vertical hourglass with two symmetric bulbs and a narrow waist.
        "hourglass": (spherical_hourglass(np.radians(24), 0.010), (mv.yz * 0.0).exp(), "#f59e0b"),
        # Parabolic bow.
        "parabola": (spherical_parabola(1.0, 0.5, 0.20), (mv.yz * 0.25).exp(), "#f97316"),
    }
    # Carry each quadric into view by pulling input points back and pushing output planes forward.
    tilted = {name: (tilt >> quadric(tilt << Point), color) for name, (quadric, tilt, color) in single.items()}
    table = {name: (stack([quadric]), [color]) for name, (quadric, color) in tilted.items()}
    trio = ("conical_donut", "lemniscate", "hourglass")
    table["trio"] = (stack([tilted[name][0] for name in trio]), [tilted[name][1] for name in trio])
    return table


def vortex_states(names: list[str], n_steps: int) -> tuple[Iterator[Quadric], list[str]]:
    """Selected preset quadrics and their colours pair with a streaming `conformal_flow` iterator over `n_steps`
    frames."""
    table = shape_catalogue()
    quadrics = concatenate([table[name][0] for name in names])                            # [n_quadrics] Plane+ <- Point-
    colors = [color for name in names for color in table[name][1]]
    return conformal_flow(quadrics, generator, n_steps), colors


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation
    from examples.quadrics.cga_spherical_quadrics import render

    states, colors = vortex_states(shape_names, n_frames)
    kept: list[Quadric] = []

    def recorded() -> Iterator[Quadric]:
        for state in states:
            kept.append(state)
            yield state

    save_animation(
        render.vortex_animation(colors, resolution, recorded()),
        f"cga_{'_'.join(shape_names)}_vortex",
        frame_duration_ms,
    )

    # checks
    # Every streamed quadric stays symmetric under polarity pullback, and stepping one more frame closes the full turn.
    initial, last = kept[0], kept[-1]
    probes = sphere_points(np.array([[0.6, 0.0, 0.8], [0.0, -0.28, 0.96], [0.36, 0.48, 0.8]]))
    assert (((last.adjugate()(probes[0, None]) & probes[1]) - (last(probes[1, None]) & probes[0])).abs() < 1e-12).all()
    one_step = (generator * (np.pi / n_frames)).exp()                                     # [] Rotor+
    closed = one_step >> last(one_step << Point)                                          # [3] Plane+ <- Point-
    assert (((closed(probes[:, None]) & probes[:, None]) - (initial(probes[:, None]) & probes[:, None])).abs() < 1e-6).all()
    carried_probes = one_step >> probes                                                   # [3] Point-
    assert ((carried_probes | carried_probes).abs() < 1e-12).all()


if __name__ == "__main__":
    main()
