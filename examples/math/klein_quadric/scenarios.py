"""Three lines on a hyperboloid of one sheet stay fixed while a steep fourth line swings into the
hyperboloid's waist and back out. Outside the waist the fourth line crosses the surface twice and both
transversals are real; through the waist it misses the surface and the transversals form a
complex-conjugate pair."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.math.klein_quadric import core

mv = core.mv
# Azimuthal angles about z where the three fixed lines cross the waist circle.
waist_angles = np.array([0.3, 2.3, 4.4])
# Rest crossing of the fourth line with the plane z = 0, its direction, and its swing amplitude along x.
fourth_crossing = [0.0, 0.4, 0.0]
fourth_direction = [0.3, -0.2, 1.0]
swing_amplitude = 1.5
# Number of rulings drawn per family on the hyperboloid.
n_rulings = 12
# Animation frame count, dense path sample count and motion share for pacing, still index, and frame delay.
n_frames = 168
n_path, line_share = 960, 0.6
still_frame = n_frames // 12
frame_duration_ms = 40


# --- math -----------------------------------------------------------------------------
def configuration(three_lines: core.Bivector,
                  fourth_line: core.Bivector) -> tuple[core.Bivector, core.Bivector, core.Point]:
    """Three fixed lines (`[3] Bivector+`) and `fourth_line` (`[...] Bivector+`) determine the
    combined four-line bundle (`[..., 4] Bivector+`), its two transversals (`[..., 2] Bivector+`), and
    its two crossings (`[..., 2] Antivector-`) with the ruled quadric of `three_lines`."""
    first, second, third = three_lines[0], three_lines[1], three_lines[2]      # [] Bivector+, Bivector+, Bivector+
    lines = stack([first.broadcast_to(fourth_line.shape), second.broadcast_to(fourth_line.shape),
                   third.broadcast_to(fourth_line.shape), fourth_line], axis=-1)   # [..., 4] Bivector+
    # Joining a point to `first`, meeting `third`, and joining to `second` yields a plane holding the point iff a transversal through it meets all three.
    surface = ((core.Point & first) ^ third) & second                          # [] Antitrivector- <- Antivector-
    return lines, core.transversals_of(lines), core.crossings_of(surface, fourth_line)


def hyperboloid_rulings(three_lines: core.Bivector, n_rulings: int) -> core.Bivector:
    """Three skew lines (`[3] Bivector+`) determine both ruling families (`[2, n_rulings] Bivector+`)
    of their ruled quadric."""
    # The unique line through a point meeting two skew lines is the meet of the two planes joining the point to each line.
    starts = core.line_points(three_lines[0], n_rulings)                       # [n_rulings] Antivector-
    transversal_family = (starts & three_lines[1]) ^ (starts & three_lines[2])   # [n_rulings] Bivector+
    # Repeating the construction across three lines of the first family recovers the family containing `three_lines`.
    second_starts = core.line_points(transversal_family[0], n_rulings)         # [n_rulings] Antivector-
    original_family = (
        (second_starts & transversal_family[n_rulings // 3])
        ^ (second_starts & transversal_family[2 * n_rulings // 3])
    )                                                                          # [n_rulings] Bivector+
    return stack([transversal_family, original_family])                        # [2, n_rulings] Bivector+


def swing_scenes(
    n_frames: int, n_rulings: int
) -> Iterator[tuple[core.Bivector, core.Bivector, core.Bivector, core.Point]]:
    """Yields per-frame `(lines, rulings, transversals, crossings)` tuples over `n_frames` as the
    fourth line swings across the hyperboloid and slows near tangency."""
    three_lines = waist_lines(waist_angles)                                    # [3] Bivector+
    # Signed crossing separation along a dense cosine path paces the swing so tangency is resolved cleanly.
    path = swing_amplitude * np.cos(np.linspace(0.0, 2 * np.pi, n_path + 1))
    _, _, path_crossings = configuration(three_lines, fourth_lines(path))      # [n_path + 1, 2] Antivector-
    separations = (path_crossings[..., 0] & path_crossings[..., 1]).scalar_norm_squared()   # [n_path + 1] Scalar+
    rulings = hyperboloid_rulings(three_lines, n_rulings)                      # [2, n_rulings] Bivector+
    for offset in paced_offsets(path, separations, n_frames, line_share):
        lines, transversals, crossings = configuration(three_lines, fourth_lines(offset))
        yield lines, rulings, transversals, crossings


# --- plumbing -------------------------------------------------------------------------
def paced_offsets(path: np.ndarray, separations: core.Scalar, n_frames: int, line_share: float) -> core.Scalar:
    """Piecewise-linear arc-length interpolation resamples `path` into `n_frames` offsets (`[n_frames]
    Scalar+`) advancing at equal combined steps of signed crossing gap and `line_share` times path
    displacement."""
    # Squared separation flips sign across tangency; its signed square root passes continuously through zero there.
    signed_squares = separations.real()                                        # [...] Scalar+
    signs = 1 - 2 * (signed_squares < 0)
    gaps = signs * (signs * signed_squares).square_root()                      # [...] Scalar+
    steps = (gaps[1:] - gaps[:-1]).abs() + line_share * np.abs(np.diff(path))  # [...] Scalar+
    pace = core.Scalar.concatenate([0.0 * steps[:1], steps.cumsum(axis=0)], axis=0)   # [...] Scalar+
    targets = pace[-1] * np.linspace(0.0, 1.0, n_frames, endpoint=False)       # [n_frames] Scalar+
    indices = np.clip((pace[:, None] <= targets[None, :]).sum(axis=0) - 1, 0, len(path) - 2)
    fractions = (targets - pace[indices]) / steps[indices]                     # [n_frames] Scalar+
    return path[indices] + fractions * (path[indices + 1] - path[indices])     # [n_frames] Scalar+


def waist_lines(angles: np.ndarray) -> core.Bivector:
    """Turning the ruling `(x - w) ^ (y - z)` about the z-axis places lines (`[...] Bivector+`) on the
    unit hyperboloid at the given waist `angles`."""
    leaning = (mv.x - mv.w) ^ (mv.y - mv.z)                                    # [] Bivector+
    turns = (mv.xy * (-angles / 2)).exp()                                      # [...] Rotor+
    return turns >> leaning                                                    # [...] Bivector+


def fourth_lines(offsets: core.Scalar) -> core.Bivector:
    """Translating the anchor along x by `offsets` yields the fourth line (`[...] Bivector+`) as the
    meet of two coordinate-aligned planes through it."""
    shifts = (mv.xw * (offsets / 2)).exp()                                     # [...] Motor+
    anchors = shifts >> mv.point(fourth_crossing)                              # [...] Antivector-
    # Joining anchor and direction gives the intrinsic line, then meeting two direction-spanned planes gives its extrinsic axis.
    intrinsic_line = anchors & mv.direction(fourth_direction)                  # [...] Antibivector-
    return (intrinsic_line & mv.direction([1, 0, 0])) ^ (intrinsic_line & mv.direction([0, 1, 0]))   # [...] Bivector+


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.math.klein_quadric import render

    recorded_scenes = []

    def streaming():
        for scene_state in swing_scenes(n_frames, n_rulings):
            recorded_scenes.append(scene_state)
            yield scene_state

    save_animation(render.swing_frames(n_frames, streaming()), "klein_swing", frame_duration_ms)
    still_fig = render.scene_figure(*recorded_scenes[still_frame])
    save_figure(still_fig, "klein_transversals")
    import matplotlib.pyplot as plt
    plt.close(still_fig)

    # checks
    lines = stack([state[0] for state in recorded_scenes])                     # [n_frames, 4] Bivector+
    rulings = recorded_scenes[0][1]                                            # [2, n_rulings] Bivector+
    transversals = stack([state[2] for state in recorded_scenes])              # [n_frames, 2] Bivector+
    crossings = stack([state[3] for state in recorded_scenes])                 # [n_frames, 2] Antivector-
    three_lines = lines[0, :3]                                                 # [3] Bivector+
    surface = ((core.Point & three_lines[0]) ^ three_lines[2]) & three_lines[1]   # [] Antitrivector- <- Antivector-
    basis_points = stack((mv.x, mv.y, mv.z, mv.w)).dual()                      # [4] Antivector-
    assert ((transversals ^ transversals).dual().abs() < 1e-11).all()
    assert (((lines[:, :, None] ^ transversals[:, None, :]).dual()).abs() < 1e-11).all()
    assert (((crossings & lines[:, 3:])[..., None] & basis_points).abs() < 1e-9).all()
    assert ((surface(crossings) & crossings).abs() < 1e-9).all()
    incidence_residuals = (                                                    # [n_frames, 2, 2] Scalar+
        ((crossings[:, :, None] & transversals[:, None, :])[..., None] & basis_points).abs().max(axis=-1)
    )
    assert (incidence_residuals.min(axis=-1) < 1e-9).all()
    ruling_samples = core.line_points(rulings, 3)                              # [2, n_rulings, 3] Antivector-
    assert ((surface(ruling_samples) & ruling_samples).abs() < 1e-9).all()


if __name__ == "__main__":
    main()
