"""Scenes for Penrose tilings slice a five-dimensional lattice patch and stream phason flips around a perpendicular loop."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.math.penrose import core

axes = core.axes
# Lattice half-span along each axis, generic cut shift with zero coordinate sum, and view half-width in edge units.
lattice_radius = 3
origin_shift = (axes * np.array([0.12, 0.29, -0.11, 0.33, -0.63])).sum()       # [] Point-
view_extent = 2.6
# Perpendicular loop radius, frame count around the loop, and frame duration in milliseconds.
phason_radius = 0.35
n_frames = 135
frame_duration_ms = 40

# Harmonic projections onto the two-dimensional cut and the perpendicular phason plane, plus the cut area element.
cut_projection = core.harmonic_projection(1)                                  # [] Point- <- Point-
phason_projection = core.harmonic_projection(2)                               # [] Point- <- Point-
cut = cut_projection(axes[0]) & cut_projection(axes[1])                       # [] Area-


def main() -> None:
    import matplotlib.pyplot as plt
    from examples.util.animation import save_animation, save_figure
    from examples.math.penrose import render

    frames = core.reciprocal_frames(cut)                                      # [10, 3] Plane+
    centres, corners = core.lattice_faces(cut_projection, lattice_radius)     # [..., 10] Point-, [..., 10, 4] Point-
    areas, volumes = core.shadow_measures(cut_projection)                     # [10] Scalar+, [10] Scalar+
    reading = core.reading_planes(cut_projection)                             # [2] Plane+
    initial_margins = core.cube_margins(origin_shift, centres, frames)        # [..., 10, 3] Scalar+
    tiling = render.tiling_figure(reading, corners, areas, view_extent, initial_margins)
    save_figure(tiling, "penrose_tiling")
    plt.close(tiling)

    loop_shifts = core.phason_loop(phason_projection, origin_shift, phason_radius, n_frames)  # [n_frames] Point-
    streamed_margins: list[core.Scalar] = []

    def recorded_margins() -> Iterator[core.Scalar]:
        for state in core.margin_stream(loop_shifts, centres, frames):
            streamed_margins.append(state)
            yield state

    save_animation(
        render.phason_frames(reading, corners, areas, view_extent, recorded_margins()),
        "penrose_phasons",
        frame_duration_ms,
    )

    # checks
    # Five applications of the axis-cycling turn return every axis to itself.
    fifth_power = list(core.cycle_powers(core.turning, 5))[-1]                # [] Point- <- Point-
    assert ((fifth_power(axes) - axes).norm() < 1e-12).all()
    # Each face's reciprocal frame pairs as the identity on its three dual-cube axes and vanishes on the cut.
    others = axes[core.other_indices]                                         # [10, 3] Point-
    frame_pairings = frames[:, None] & others[..., None]                      # [10, 3, 3] Scalar+
    assert ((frame_pairings - np.eye(3)).abs() < 1e-12).all()
    assert ((frames & cut_projection(axes)[:, None, None]).abs() < 1e-12).all()
    # Dual-cube shadow volumes equal projected tile areas, and thick-to-thin areas and patch counts follow the golden ratio.
    golden = (1 + np.sqrt(5)) / 2
    assert ((volumes - areas).abs() < 1e-12).all()
    thick = areas > areas.mean()                                              # [10]
    assert ((areas[thick] / areas[~thick][:, None] - golden).abs() < 1e-12).all()
    selected = np.all(initial_margins > 0, axis=-1)                           # [..., 10]
    assert abs(selected[:, thick].sum() / selected[:, ~thick].sum() - golden) < 5e-2 * golden
    assert len(streamed_margins) == n_frames
    assert ((streamed_margins[0] - initial_margins).abs() < 1e-12).all()


if __name__ == "__main__":
    main()
