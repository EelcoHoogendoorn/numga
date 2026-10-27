"""Scenes for the earthquake beachballs: four kinds of fault side by side, and a fault whose slip
turns through every rake."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.mechanics.earthquake import core

north, east, down = core.north, core.east, core.down
# The faults side by side, strike-slip, normal, thrust and oblique: their strikes, dips and rakes, in degrees.
STRIKES = np.array([30.0, 30.0, 30.0, 30.0])
DIPS = np.array([90.0, 60.0, 30.0, 50.0])
RAKES = np.array([0.0, -90.0, 90.0, 45.0])
# The seismic moment of every fault.
MOMENT = 1.0
# The sweep: one fault's strike and dip, the frames of its rake's turn, and the duration of each.
SWEEP_STRIKE, SWEEP_DIP = 30.0, 50.0
N_FRAMES = 72
DURATION_MS = 80
# The grid of directions: rings from straight down, spokes around the vertical.
N_RINGS = 60
N_SPOKES = 120


# --- math -----------------------------------------------------------------------------
def faults(strikes: np.ndarray, dips: np.ndarray, rakes: np.ndarray) -> tuple[core.Plane, core.Plane, core.Moment]:
    """The fault plane, auxiliary plane and moment of each fault of the given strike, dip and rake, in
    radians."""
    frames = core.fault_frame(strikes, dips, rakes)                           # [...] Rotor+
    planes, auxiliaries = frames >> core.ground, frames >> core.prime_vertical   # [...] Plane+, Plane+
    return planes, auxiliaries, core.double_couple(planes, auxiliaries, MOMENT)


def beachballs(moment: core.Moment) -> tuple[core.Scalar, core.Point]:
    """Each fault's P wave over the lower hemisphere `[faults, rings, spokes]`, and its principal axes
    `[faults, 3]`, from pressure to tension."""
    _, axes = core.principal_axes(moment)                                     # [faults, 3] Point-
    push, _ = core.radiation(moment[:, None, None], hemisphere())             # [faults, rings, spokes] Scalar+
    return push, axes


def sweep(n_frames: int) -> Iterator[tuple[core.Point, core.Point, core.Point, core.Scalar, core.Point, core.Scalar]]:
    """Per frame of one fault's rake turning once around: its normal, slip and principal axes, its P wave
    over the lower hemisphere, and the whole sphere of directions in the fault's frame with the P wave
    over it."""
    for rake in np.linspace(-np.pi, np.pi, n_frames, endpoint=False):
        frame = core.fault_frame(np.deg2rad(SWEEP_STRIKE), np.deg2rad(SWEEP_DIP), np.array(rake))   # [] Rotor+
        fault, auxiliary = frame >> core.ground, frame >> core.prime_vertical   # [] Plane+, Plane+
        moment = core.double_couple(fault, auxiliary, MOMENT)                 # [] Moment
        _, axes = core.principal_axes(moment)                                 # [3] Point-
        # The metric turns the two planes into the fault's normal and its slip, for drawing.
        normal, slip = fault.dual(), auxiliary.dual()                         # [] Point-, Point-
        # The sphere turned with the fault, so that its nodal planes lie along the grid lines.
        fault_sphere = frame >> sphere()                                      # [rings, spokes] Point-
        ball, _ = core.radiation(moment, hemisphere())                        # [rings, spokes] Scalar+
        lobes, _ = core.radiation(moment, fault_sphere)                       # [rings, spokes] Scalar+
        yield normal, slip, axes, ball, fault_sphere, lobes


# --- plumbing -------------------------------------------------------------------------
def hemisphere() -> core.Point:
    """The lower half of the sphere of directions, where seismologists draw the beachball."""
    return core.directions(np.linspace(0.0, np.pi / 2, N_RINGS), np.linspace(0.0, 2 * np.pi, N_SPOKES + 1))   # [rings, spokes + 1] Point-


def sphere() -> core.Point:
    """The whole sphere of directions, with the hemisphere's spacing and a ring on its equator."""
    return core.directions(np.linspace(0.0, np.pi, 2 * N_RINGS - 1), np.linspace(0.0, 2 * np.pi, N_SPOKES + 1))   # [2 * rings - 1, spokes + 1] Point-


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.mechanics.earthquake import render

    planes, auxiliaries, moment = faults(np.deg2rad(STRIKES), np.deg2rad(DIPS), np.deg2rad(RAKES))   # [faults]
    push, axes = beachballs(moment)
    save_figure(render.draw_mechanisms(hemisphere(), push, axes), "earthquake_beachballs")
    save_animation(render.animate_sweep(hemisphere(), sweep(N_FRAMES)), "earthquake_rake", DURATION_MS)

    # checks
    # A slip neither swells nor shrinks the ground and has no third axis: its principal values are minus
    # the moment, zero and the moment, so their sum, the trace, and the moment's determinant against the
    # metric vanish.
    values, _ = core.principal_axes(moment)                                   # [faults, 3] Scalar+
    np.testing.assert_allclose(values.sum(axis=-1).to_array(), 0.0, atol=1e-14)
    np.testing.assert_allclose((moment(core.Point) & core.Point).det(core.Point.dual().dot(core.Point.dual())).to_array(), 0.0, atol=1e-14)
    np.testing.assert_allclose(values.to_array(), np.array([-MOMENT, 0.0, MOMENT]) * np.ones((len(STRIKES), 1)), atol=1e-13)
    # The slip's couple alone would turn the ground, by the plane of shear; the double couple does not.
    single = auxiliaries * (planes & core.Point)                              # [faults] Moment
    np.testing.assert_allclose(core.curl(single).kernel, (planes ^ auxiliaries).kernel, atol=1e-14)
    np.testing.assert_allclose(core.curl(moment).kernel, 0.0, atol=1e-14)
    # The fault and the auxiliary plane give the same moment.
    swapped = core.double_couple(auxiliaries, planes, MOMENT)                 # [faults] Moment
    np.testing.assert_allclose(swapped.kernel, moment.kernel, atol=1e-14)
    # From the moment back to the fault: both planes rebuild the moment, and one of them is the fault.
    found, found_auxiliaries = core.fault_planes(moment)                      # [faults, 2] Plane+, Plane+
    rebuilt = core.double_couple(found, found_auxiliaries, MOMENT)            # [faults, 2] Moment
    np.testing.assert_allclose(rebuilt.kernel, moment[:, None].broadcast_to(rebuilt.shape).kernel, atol=1e-13)
    alignment = found.dot(planes[:, None]).abs().to_array()                   # [faults, 2]
    np.testing.assert_allclose(alignment.max(axis=-1), 1.0, rtol=1e-12)
    # The P wave is strongest along the tension axis, and silent across the fault and along the slip.
    axis_push, _ = core.radiation(moment[:, None], axes)                      # [faults, 3] Scalar+
    np.testing.assert_allclose(axis_push.to_array(), values.to_array(), atol=1e-13)
    # The slip is the direction across the auxiliary plane.
    silent, _ = core.radiation(moment, auxiliaries.dual())                    # [faults] Scalar+
    np.testing.assert_allclose(silent.to_array(), 0.0, atol=1e-14)
    # Turning the rake turns the slip within the fault, and half a turn reverses the slip and flips the moment.
    rakes = np.linspace(-np.pi, np.pi, N_FRAMES, endpoint=False)               # [frames]
    sweep_planes, sweep_auxiliaries, sweep_moment = faults(np.deg2rad(SWEEP_STRIKE), np.deg2rad(SWEEP_DIP), rakes)
    np.testing.assert_allclose((sweep_planes - sweep_planes[0]).norm().to_array(), 0.0, atol=1e-14)
    # The slip, across the auxiliary plane, lies in the fault.
    np.testing.assert_allclose((sweep_planes & sweep_auxiliaries.dual()).to_array(), 0.0, atol=1e-14)
    half = N_FRAMES // 2
    np.testing.assert_allclose((sweep_moment[half:] + sweep_moment[:half]).kernel, 0.0, atol=1e-14)


if __name__ == "__main__":
    main()
