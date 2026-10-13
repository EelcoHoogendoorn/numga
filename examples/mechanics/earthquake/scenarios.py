"""Scenes for the earthquake beachballs: four kinds of fault side by side, and a fault whose slip
turns through every rake."""

from __future__ import annotations

import numpy as np

from examples.mechanics.earthquake import core

north, east, down = core.north, core.east, core.down
# The faults side by side: their names, strikes, dips and rakes, in degrees.
NAMES = ["strike-slip", "normal", "thrust", "oblique"]
STRIKES = np.array([30.0, 30.0, 30.0, 30.0])
DIPS = np.array([90.0, 60.0, 30.0, 50.0])
RAKES = np.array([0.0, -90.0, 90.0, 45.0])
# The seismic moment of every fault.
MOMENT = 1.0
# The sweep: one fault's strike and dip, and the frames of its rake's turn.
SWEEP_STRIKE, SWEEP_DIP = 30.0, 50.0
FRAMES = 72
# The grid of directions: rings from straight down, spokes around the vertical.
RINGS = 60
SPOKES = 120


# --- math -----------------------------------------------------------------------------
def hemisphere():
    """The lower half of the sphere of directions, where seismologists draw the beachball."""
    return core.directions(np.linspace(0.0, np.pi / 2, RINGS), np.linspace(0.0, 2 * np.pi, SPOKES + 1))   # [RINGS, SPOKES + 1] Vector


def sphere():
    """The whole sphere of directions."""
    return core.directions(np.linspace(0.0, np.pi, RINGS), np.linspace(0.0, 2 * np.pi, SPOKES + 1))       # [RINGS, SPOKES + 1] Vector


def mechanisms() -> core.Mechanism:
    """The four faults, with their first motions over the lower hemisphere."""
    frames = core.fault_frame(np.deg2rad(STRIKES), np.deg2rad(DIPS), np.deg2rad(RAKES))   # [faults] Rotor
    faults = core.mechanism(frames, MOMENT, hemisphere())

    # --- checks
    # A slip neither swells nor shrinks the ground and has no third axis: the moment's trace and
    # determinant vanish, and its principal values are minus the moment, zero and the moment.
    np.testing.assert_allclose(faults.moment.trace().to_array(), 0.0, atol=1e-14)
    np.testing.assert_allclose(faults.moment.det().to_array(), 0.0, atol=1e-14)
    np.testing.assert_allclose(faults.values.to_array(), np.array([-MOMENT, 0.0, MOMENT]) * np.ones((len(NAMES), 1)), atol=1e-13)
    # The slip's couple alone would turn the ground, by the plane of shear; the double couple does not.
    single = faults.slips * (faults.normals | core.Vector)                    # [faults] Moment
    np.testing.assert_allclose(core.curl(single).kernel, (faults.normals ^ faults.slips).kernel, atol=1e-14)
    np.testing.assert_allclose(core.curl(faults.moment).kernel, 0.0, atol=1e-14)
    # The fault and the auxiliary plane give the same moment.
    swapped = core.double_couple(faults.slips, faults.normals, MOMENT)       # [faults] Moment
    np.testing.assert_allclose(swapped.kernel, faults.moment.kernel, atol=1e-14)
    # From the moment back to the fault: both planes rebuild the moment, and one of them is the fault.
    normals, slips = core.fault_planes(faults.moment)                         # [faults, 2] Vector each
    rebuilt = core.double_couple(normals, slips, MOMENT)                      # [faults, 2] Moment
    np.testing.assert_allclose(rebuilt.kernel, faults.moment[:, None].broadcast_to(rebuilt.shape).kernel, atol=1e-13)
    alignment = (normals | faults.normals[:, None]).abs().to_array()          # [faults, 2]
    np.testing.assert_allclose(alignment.max(axis=-1), 1.0, rtol=1e-12)
    # The P wave is strongest along the tension axis, and silent across the fault and along the slip.
    push, _ = core.radiation(faults.moment[:, None], faults.axes)             # [faults, 3] Scalar
    np.testing.assert_allclose(push.to_array(), faults.values.to_array(), atol=1e-13)
    silent, _ = core.radiation(faults.moment, faults.slips)                   # [faults] Scalar
    np.testing.assert_allclose(silent.to_array(), 0.0, atol=1e-14)
    return faults


def sweep() -> tuple[core.Mechanism, core.Mechanism]:
    """One fault whose rake turns once around: its first motions over the lower hemisphere, and over
    the whole sphere."""
    rakes = np.linspace(-np.pi, np.pi, FRAMES, endpoint=False)
    frames = core.fault_frame(np.deg2rad(SWEEP_STRIKE), np.deg2rad(SWEEP_DIP), rakes)   # [FRAMES] Rotor
    balls, lobes = core.mechanism(frames, MOMENT, hemisphere()), core.mechanism(frames, MOMENT, sphere())

    # --- checks
    # Turning the rake turns the slip within the fault: the normal stays, the slip stays in the plane.
    np.testing.assert_allclose((balls.normals - balls.normals[0]).norm().to_array(), 0.0, atol=1e-14)
    np.testing.assert_allclose((balls.slips | balls.normals).to_array(), 0.0, atol=1e-14)
    # Half a turn of rake reverses the slip, and leaves the moment's sign flipped.
    half = FRAMES // 2
    np.testing.assert_allclose((balls.moment[half:] + balls.moment[:half]).kernel, 0.0, atol=1e-14)
    return balls, lobes


if __name__ == "__main__":
    from examples.animation import save_animation, save_figure
    from examples.mechanics.earthquake import render

    save_figure(render.draw_mechanisms(mechanisms(), NAMES), "earthquake_beachballs")
    save_animation(render.animate_sweep(*sweep()), "earthquake_rake", 80)
