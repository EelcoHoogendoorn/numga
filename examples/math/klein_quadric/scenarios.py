"""The scenes of the four-line problem hold three lines on the hyperboloid x² + y² = 1 + z² and a
steep fourth line swinging across it, through its waist and out the other side. Outside, the fourth
line crosses the hyperboloid twice and both transversals are real; through the waist it misses the
hyperboloid and the transversals are a complex-conjugate pair."""

from __future__ import annotations

from collections.abc import Iterator
from itertools import islice

import numpy as np

from numga import stack
from examples.math.klein_quadric import core

mv = core.mv
# Where the three lines cross the hyperboloid's waist, by angle about z.
waist_angles = np.array([0.3, 2.3, 4.4])
# The fourth line: where it crosses the plane z = 0 at rest, the direction it runs in, and how far it
# swings along x either way.
fourth_crossing = [0.0, 0.4, 0.0]
fourth_direction = [0.3, -0.2, 1.0]
swing_amplitude = 1.5
# The lines drawn of each family on the hyperboloid.
n_rulings = 12
# The frames of the swing; the points it is sampled at to pace it, and the share of the line's own motion in the
# pace; the frame shown as a still, and the duration of a frame in milliseconds.
n_frames = 168
n_path, line_share = 960, 0.6
still_frame = n_frames // 12
frame_duration_ms = 40


# --- math -----------------------------------------------------------------------------
def scene(three: core.Bivector, fourth: core.Bivector) -> tuple[core.Bivector, core.Bivector, core.Point]:
    """The four lines, the two transversals of all four and the two points where the fourth line
    crosses the quadric of the three, for each of a batch of fourth lines."""
    first, second, third = three[0], three[1], three[2]                        # [] Bivector+, Bivector+, Bivector+
    lines = stack([first.broadcast_to(fourth.shape), second.broadcast_to(fourth.shape),
                   third.broadcast_to(fourth.shape), fourth], axis=-1)         # [..., 4] Bivector+
    # The plane joining a point to the first line meets the third line in a point; the plane joining
    # that point to the second line holds the first point exactly when a line through it meets all three.
    surface = ((core.Point & first) ^ third) & second                          # [] Plane- <- Point-
    return lines, core.transversals_of(lines), core.crossings_of(surface, fourth)


def rulings(three: core.Bivector, n_rulings: int) -> core.Bivector:
    """Both families of lines on the quadric of three lines: the lines through points spread over the
    first that meet the other two, and the lines that meet three of those, the family of the three."""
    # The line through a point that meets two lines is the meet of the planes joining the point to each.
    starts = core.spread(three[0], n_rulings)                                  # [rulings] Point-
    sweep = (starts & three[1]) ^ (starts & three[2])                          # [rulings] Bivector+
    family_starts = core.spread(sweep[0], n_rulings)                           # [rulings] Point-
    family = (family_starts & sweep[n_rulings // 3]) ^ (family_starts & sweep[2 * n_rulings // 3])   # [rulings] Bivector+
    return stack([sweep, family])                                              # [families, rulings] Bivector+


def swing(n_frames: int, n_rulings: int) -> Iterator[tuple[core.Bivector, core.Bivector, core.Bivector, core.Point]]:
    """The fourth line swung once across the hyperboloid and back, slowing where its crossings merge: per frame the
    four lines, both families of lines on the hyperboloid, the transversals and the crossings."""
    three = waist_lines(waist_angles)                                          # [3] Bivector+
    # The swing sampled closely along a closed path, and the squared distance between the crossings along it,
    # negative where they are a complex pair.
    path = swing_amplitude * np.cos(np.linspace(0.0, 2 * np.pi, n_path + 1))
    _, _, along_path = scene(three, steep(path))                               # [path + 1, 2] Point-
    separations = (along_path[..., 0] & along_path[..., 1]).scalar_norm_squared()   # [path + 1] Scalar+
    hyperboloid = rulings(three, n_rulings)                                    # [families, rulings] Bivector+
    for offset in paced(path, separations, n_frames, line_share):
        lines, transversals, crossings = scene(three, steep(offset))
        yield lines, hyperboloid, transversals, crossings


# --- plumbing -------------------------------------------------------------------------
def paced(path: np.ndarray, separations: core.Scalar, n_frames: int, line_share: float) -> np.ndarray:
    """Offsets along a closed path, at equal steps of the gap between the fourth line's crossings and of a share of
    the line's own motion: slow where the crossings merge, so the transversals are seen to close. The gap is the
    distance between the crossings while they are real and minus it once they are a complex pair, where their
    squared separation turns negative, so that it passes through zero where they merge."""
    squared = separations.to_array().real
    gaps = np.sign(squared) * np.sqrt(np.abs(squared))
    pace = np.concatenate([[0.0], np.cumsum(np.abs(np.diff(gaps)) + line_share * np.abs(np.diff(path)))])
    return np.interp(np.linspace(0.0, pace[-1], n_frames, endpoint=False), pace, path)


def waist_lines(angles: np.ndarray) -> core.Bivector:
    """Lines of the hyperboloid x² + y² = 1 + z² through the points of its waist at the given angles,
    all leaning the same way."""
    # The line through (1, 0, 0) leaning along (0, 1, 1), the meet of the planes x = 1 and y = z, turned
    # about z by each angle.
    leaning = (mv.x - mv.w) ^ (mv.y - mv.z)                                    # [] Bivector+
    turns = (mv.xy * (-angles / 2)).exp()                                      # [angles] Rotor+
    return turns >> leaning                                                    # [angles] Bivector+


def steep(offsets: np.ndarray) -> core.Bivector:
    """The fourth line, moved along x by each offset: the meet of the two planes through it along x
    and along y."""
    shifts = (mv.xw * (offsets / 2)).exp()                                     # [offsets] Motor+
    anchors = shifts >> mv.point(fourth_crossing)                              # [offsets] Point-
    # The fourth line as the join of its anchor and its direction, and again as a meet of planes.
    line = anchors & mv.direction(fourth_direction)                            # [offsets] Line-
    return (line & mv.direction([1, 0, 0])) ^ (line & mv.direction([0, 1, 0]))   # [offsets] Bivector+


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.math.klein_quadric import render

    still = next(islice(swing(n_frames, n_rulings), still_frame, None))
    save_figure(render.draw(*still), "klein_transversals")
    save_animation(render.animate(swing(n_frames, n_rulings), n_frames), "klein_swing", frame_duration_ms)

    # checks
    # The transversals are lines and meet all four lines; the crossings lie on the fourth line and on
    # the quadric, each on one of the transversals; both families lie on the quadric.
    three = waist_lines(waist_angles)
    fourth = steep(swing_amplitude * np.cos(np.linspace(0.0, 2 * np.pi, n_frames, endpoint=False)))
    lines, transversals, crossings = scene(three, fourth)
    surface = ((core.Point & three[0]) ^ three[2]) & three[1]
    np.testing.assert_allclose((transversals ^ transversals).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose((lines[..., :, None] ^ transversals[..., None, :]).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose((crossings & lines[..., 3:]).kernel, 0.0, atol=1e-10)
    np.testing.assert_allclose((surface(crossings) & crossings).kernel, 0.0, atol=1e-10)
    off = np.abs((crossings[..., :, None] & transversals[..., None, :]).kernel).max(axis=-1)   # [frames, 2, 2]
    np.testing.assert_allclose(off.min(axis=-1), 0.0, atol=1e-10)
    ends = core.spread(rulings(three, n_rulings), 3)                           # [families, rulings, 3] Point-
    np.testing.assert_allclose((surface(ends) & ends).kernel, 0.0, atol=1e-10)


if __name__ == "__main__":
    main()
