"""Balanced four-ply layups under axial tension compare an antisymmetric stack that twists against
a symmetric stack that stays flat."""

from collections.abc import Iterator

import numpy as np

from examples.mechanics.composite_strip import core

angles = np.deg2rad([[-45, -45, 45, 45], [-45, 45, 45, -45]])                  # [2, 4], bottom to top
longitudinal_modulus = 130000.0                                                # MPa
transverse_modulus = 10000.0                                                   # MPa
poisson = 0.3
shear_modulus = 5000.0                                                         # MPa
thickness = 1.0                                                                # mm
length = 100.0                                                                 # mm
width = 20.0                                                                   # mm
tension = 30.0                                                                 # N/mm of width
n_length_samples = 61
n_width_samples = 13
magnification = 5.0
fibre_spacing = 5.0                                                            # mm between drawn fibre lines
separation = 1.6 * width                                                       # mm between strip centrelines
n_frames = 90
duration_ms = 40
# The pull grows smoothly from rest to full load and falls back over one period.
loads = (1 - np.cos(np.linspace(0, 2 * np.pi, n_frames, endpoint=False))) / 2   # [n_frames]


# --- math -----------------------------------------------------------------------------
def strip_pull() -> tuple[core.State, core.Scalar, core.Direction, core.Direction, core.Scalar, Iterator[core.Direction]]:
    """Solving both four-ply layups under `tension` yields their extension-twist states (`[2] State+`),
    width strains (`[2] Scalar+`), rest and peak-load midsurface grids (`[2, n_length_samples, n_width_samples] Direction-`),
    top-ply transverse coordinates (`[2, n_length_samples, n_width_samples] Scalar+`), and the streaming
    pull frames."""
    n_plies = angles.shape[-1]
    n_nodes = 2
    ply_thickness = thickness / n_plies
    # Two Gauss nodes per ply integrate its quadratic strain energy exactly.
    nodes = np.array([-1, 1]) / np.sqrt(3)                                     # [2]
    centres = (np.arange(n_plies) + 0.5) * ply_thickness - thickness / 2       # [n_plies]
    heights = core.unit_z * (centres[:, None] + nodes * ply_thickness / 2)     # [n_plies, n_nodes] Direction-
    weights = np.full((n_plies, n_nodes), ply_thickness / 2)                   # [n_plies, n_nodes]
    # Rotate each ply's fibre direction in the strip plane; both layups share the thickness nodes.
    ply_rotors = (core.mv.xy * (-angles[..., None] / 2)).exp()                 # [2, n_plies, 1] Rotor+
    fibres = ply_rotors >> core.unit_x                                         # [2, n_plies, 1] Direction-
    state, width_strain = core.tension_response(
        fibres, heights, weights, thickness,
        longitudinal_modulus, transverse_modulus, poisson, shear_modulus, tension,
    )                                                                          # [2] State+, [2] Scalar+
    # One flat reference midsurface grid serves both strips before side-by-side placement.
    reference = (
        core.unit_x * np.linspace(0, length, n_length_samples)[:, None]
        + core.unit_y * np.linspace(-width / 2, width / 2, n_width_samples)[None, :]
    )                                                                          # [n_length_samples, n_width_samples] Direction-
    placement = core.unit_y * (separation * np.array([-0.5, 0.5]))[:, None, None]  # [2, 1, 1] Direction-
    rest = reference + placement                                               # [2, n_length_samples, n_width_samples] Direction-
    peak = core.midsurface_points(
        reference, state * magnification, width_strain * magnification, thickness,
    ) + placement                                                              # [2, n_length_samples, n_width_samples] Direction-
    # Level lines of the coordinate perpendicular to the top ply's fibres trace those fibres on each strip.
    top_normal = ply_rotors[:, -1, 0] >> core.mv.y                             # [2] Plane+
    across_fibres = top_normal[:, None, None] & reference                      # [2, n_length_samples, n_width_samples] Scalar+
    frames = core.pull_frames(reference, placement, state, width_strain, thickness, magnification * loads)
    return state, width_strain, rest, peak, across_fibres, frames


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.mechanics.composite_strip import render
    from examples.util.animation import save_animation

    state, width_strain, rest, peak, across_fibres, frames = strip_pull()
    recorded_frames: list[core.Direction] = []

    def streaming_frames() -> Iterator[core.Direction]:
        for grid in frames:
            recorded_frames.append(grid)
            yield grid

    save_animation(
        render.animate_strips(rest, peak, across_fibres, fibre_spacing, streaming_frames()),
        "composite_strip", duration_ms,
    )

    # checks
    assert len(recorded_frames) == n_frames
    assert ((recorded_frames[0] - rest).antinorm() < 1e-12).all()
    assert ((recorded_frames[n_frames // 2] - peak).antinorm() < 1e-12).all()
    assert (core.extension(state) > 0).all()
    assert (width_strain < 0).all()
    assert core.twist(state[0]).abs() > 1e-3
    assert core.twist(state[1]).abs() < 1e-12
    assert core.extension(state[0]) > core.extension(state[1])


if __name__ == "__main__":
    main()
