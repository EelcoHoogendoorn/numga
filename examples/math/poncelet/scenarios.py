"""Two-panel Poncelet porism scene comparing an inner envelope tangent to an inscribed pentagon
against a second inner envelope tangent to a slightly scaled pentagon."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.math.poncelet import core

# Ellipsoid centre, semi-axes along its principal directions, and tilt angle from `x`.
centre = np.array([0.4, -0.2])                                                 # [2]
semi_axes = np.array([2.0, 1.2])                                               # [2]
tilt = 0.5
# Parameter angles of the five pentagon vertices, cyclic next-vertex indices, and radial scales of the two pentagons.
corner_angles = np.array([0.1, 0.8, 1.9, 3.0, 5.2])                            # [5]
next_corners = np.array([1, 2, 3, 4, 0])                                       # [5]
pentagon_scales = np.array([1.0, 0.97])                                        # [2]
# Polygon side count, animation frame count, starting angle offset, static-figure frame index, frame delay, and palette size.
n_sides = 5
n_frames = 180
start_offset = 0.05
still_frame = n_frames // 12
frame_duration_ms = 40
n_colours = 32
# Rigid motor turning the ellipsoid by `tilt` and translating its origin to `centre`.
centre_shift = ((core.mv.xw * centre[0] + core.mv.yw * centre[1]) * 0.5).exp() # [] Motor+
tilt_turn = (core.mv.xy * (-tilt / 2)).exp()                                   # [] Motor+
placement = centre_shift * tilt_turn                                           # [] Motor+


# --- math -----------------------------------------------------------------------------
def scene() -> tuple[core.Quadric, core.Envelope, core.Point]:
    """An inscribed pentagon and a scaled pentagon inside the tilted ellipsoid define the outer
    quadric (`[] Plane+ <- Point-`), the two inner envelopes (`[2] Pole+ <- Line-`), and their
    shared interior point (`[] Point-`)."""
    outer = core.ellipsoid(semi_axes, placement)                               # [] Plane+ <- Point-
    unit_corners = np.stack([np.cos(corner_angles), np.sin(corner_angles)], axis=-1)   # [5, 2]
    corners = placement >> core.mv.point(pentagon_scales[:, None, None] * semi_axes * unit_corners)   # [2, 5] Point-
    inner = core.envelope(corners & corners[:, next_corners])                  # [2] Pole+ <- Line-
    inside = core.mv.point(centre)                                             # [] Point-
    return outer, inner, inside


def circuit_paths(outer: core.Quadric, inner: core.Envelope, inside: core.Point,
                  n_starts: int) -> Iterator[core.Point]:
    """Yields the `[n_sides + 1, 2] Point-` paths around both inner envelopes for `n_starts`
    evenly spaced parameter angles around `outer`."""
    start_angles = np.linspace(0.0, 2 * np.pi, n_starts, endpoint=False) + start_offset   # [n_starts]
    unit_starts = np.stack([np.cos(start_angles), np.sin(start_angles)], axis=-1)         # [n_starts, 2]
    starts = placement >> core.mv.point(semi_axes * unit_starts)               # [n_starts] Point-
    return core.polygon_paths(outer, inner, starts, inside, n_sides)


def main() -> None:
    import matplotlib.pyplot as plt

    from examples.util.animation import save_animation, save_figure
    from examples.math.poncelet import render

    outer, inner, inside = scene()                                             # [] Plane+ <- Point-, [2] Pole+ <- Line-, [] Point-
    recorded_paths: list[core.Point] = []

    def path_stream() -> Iterator[core.Point]:
        for frame_vertices in circuit_paths(outer, inner, inside, n_frames):
            recorded_paths.append(frame_vertices)
            yield frame_vertices

    save_animation(render.path_frames(outer, inner, path_stream()), "poncelet_turn", frame_duration_ms, n_colours)
    vertices = stack(recorded_paths)                                           # [n_frames, n_sides + 1, 2] Point-
    still_figure = render.paths_figure(outer, inner, vertices[still_frame])
    save_figure(still_figure, "poncelet_pentagons")
    plt.close(still_figure)

    # checks
    # Every vertex lies on the outer ellipsoid, every side touches its inner envelope, and five-sided paths
    # close from every starting point around the first inner envelope while missing around the second.
    assert ((outer(vertices) & vertices).abs() < 1e-10).all()
    sides = (vertices[:, :-1] & vertices[:, 1:]).normalized()                  # [n_frames, n_sides, 2] Line-
    horizon = core.mv.x.dual() & core.mv.y.dual()                              # [] Line-
    touch = (inner(sides) & sides) / (inner(horizon) & horizon)                # [n_frames, n_sides, 2] Scalar+
    assert (touch.abs() < 1e-10).all()
    gaps = (vertices[:, n_sides] - vertices[:, 0]).antinorm()                  # [n_frames, 2] Scalar+
    assert (gaps[:, 0] < 1e-9).all()
    assert (gaps[:, 1] > 1e-2).all()


if __name__ == "__main__":
    main()
