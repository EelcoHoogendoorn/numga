"""Scene parameters and rake-sweep generator for earthquake focal mechanism beachballs and radiation lobes."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.mechanics.earthquake import core

# Strike-slip, normal, thrust and oblique faults: strikes, dips and rakes in degrees, plus unit seismic moment.
strikes = np.array([30.0, 30.0, 30.0, 30.0])
dips = np.array([90.0, 60.0, 30.0, 50.0])
rakes = np.array([0.0, -90.0, 90.0, 45.0])
seismic_moment = 1.0
# Rake-sweep parameters: fixed strike and dip in degrees, frame count, and frame duration in milliseconds.
sweep_strike, sweep_dip = 30.0, 50.0
n_frames = 144
frame_duration_ms = 40
# Spherical direction grid counts: polar rings from straight down and azimuthal spokes around the vertical.
n_rings = 60
n_spokes = 120


def fault_mechanisms(fault_strikes: np.ndarray, fault_dips: np.ndarray, fault_rakes: np.ndarray) -> tuple[core.Plane, core.Plane, core.Moment]:
    """Orienting faults by `fault_strikes`, `fault_dips` and `fault_rakes` (`[...]`, in radians) gives their fault planes, auxiliary planes (`[...] Plane+`), and double-couple moments (`[...] Plane+ <- Point-`)."""
    frames = core.fault_frame(fault_strikes, fault_dips, fault_rakes)         # [...] Rotor+
    planes, auxiliaries = frames >> core.ground, frames >> core.prime_vertical   # [...] Plane+, [...] Plane+
    return planes, auxiliaries, core.double_couple(planes, auxiliaries, seismic_moment)


def beachballs(moment: core.Moment) -> tuple[core.Scalar, core.Point]:
    """Evaluating `moment` (`[...] Plane+ <- Point-`) over the lower hemisphere yields P-wave amplitudes (`[..., n_rings, n_spokes + 1] Scalar+`) and principal axes (`[..., 3] Point-`)."""
    _, axes = core.principal_axes(moment)                                     # [..., 3] Point-
    push, _ = core.radiation(moment[..., None, None], hemisphere())           # [..., n_rings, n_spokes + 1] Scalar+
    return push, axes


def rake_sweep(n_frames: int) -> Iterator[tuple[core.Point, core.Point, core.Point, core.Scalar, core.Point, core.Scalar, core.Moment]]:
    """Yields `(normal, slip, axes, ball, fault_sphere, lobes, moment)` across `n_frames` evenly spaced rakes over a full turn on a fixed fault plane."""
    unit_hemisphere = hemisphere()                                            # [n_rings, n_spokes + 1] Point-
    unit_sphere = sphere()                                                    # [2 * n_rings - 1, n_spokes + 1] Point-
    for rake in np.linspace(-np.pi, np.pi, n_frames, endpoint=False):
        frame = core.fault_frame(np.deg2rad(sweep_strike), np.deg2rad(sweep_dip), np.array(rake))   # [] Rotor+
        fault, auxiliary = frame >> core.ground, frame >> core.prime_vertical   # [] Plane+, [] Plane+
        moment = core.double_couple(fault, auxiliary, seismic_moment)         # [] Plane+ <- Point-
        _, axes = core.principal_axes(moment)                                 # [3] Point-
        # Dualizing the fault and auxiliary planes gives the unit normal and slip direction for 3D arrows.
        normal, slip = fault.dual(), auxiliary.dual()                         # [] Point-, [] Point-
        # Rotating the sampling sphere with the fault aligns grid lines cleanly with the nodal planes.
        fault_sphere = frame >> unit_sphere                                   # [2 * n_rings - 1, n_spokes + 1] Point-
        ball, _ = core.radiation(moment, unit_hemisphere)                     # [n_rings, n_spokes + 1] Scalar+
        lobes, _ = core.radiation(moment, fault_sphere)                       # [2 * n_rings - 1, n_spokes + 1] Scalar+
        yield normal, slip, axes, ball, fault_sphere, lobes, moment


# --- plumbing -------------------------------------------------------------------------
def hemisphere() -> core.Point:
    """Samples unit directions (`[n_rings, n_spokes + 1] Point-`) over the lower hemisphere."""
    return core.sphere_directions(np.linspace(0.0, np.pi / 2, n_rings), np.linspace(0.0, 2 * np.pi, n_spokes + 1))   # [n_rings, n_spokes + 1] Point-


def sphere() -> core.Point:
    """Samples unit directions (`[2 * n_rings - 1, n_spokes + 1] Point-`) over the full sphere with a ring on the equator."""
    return core.sphere_directions(np.linspace(0.0, np.pi, 2 * n_rings - 1), np.linspace(0.0, 2 * np.pi, n_spokes + 1))   # [2 * n_rings - 1, n_spokes + 1] Point-


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.mechanics.earthquake import render

    planes, auxiliaries, moment = fault_mechanisms(np.deg2rad(strikes), np.deg2rad(dips), np.deg2rad(rakes))   # [4] Plane+, [4] Plane+, [4] Plane+ <- Point-
    values, axes = core.principal_axes(moment)                                # [4, 3] Scalar+, [4, 3] Point-
    lower_hemisphere = hemisphere()                                           # [n_rings, n_spokes + 1] Point-
    push, _ = core.radiation(moment[:, None, None], lower_hemisphere)         # [4, n_rings, n_spokes + 1] Scalar+
    save_figure(render.draw_mechanisms(lower_hemisphere, push, axes), "earthquake_beachballs")

    states = []

    def streamed():
        for state in rake_sweep(n_frames):
            states.append(state)
            yield state

    save_animation(render.animate_sweep(lower_hemisphere, streamed()), "earthquake_rake", frame_duration_ms)

    # checks
    # Pure shear slip neither dilates nor compresses the rock and leaves the null axis unstressed, so eigenvalues are (-M_0, 0, M_0).
    expected_values = np.array([-seismic_moment, 0.0, seismic_moment])
    assert ((values - expected_values).abs() < 1e-12).all()
    assert ((moment(core.Point) & core.Point).det(core.Point.dot(core.Point)).abs() < 1e-12).all()
    # A single couple exerts net shear torque `planes ^ auxiliaries`, whereas the double couple is curl-free.
    single = auxiliaries * (planes & core.Point)                              # [4] Plane+ <- Point-
    assert ((core.curl(single) - (planes ^ auxiliaries)).norm() < 1e-12).all()
    assert (core.curl(moment).norm() < 1e-12).all()
    # Exchanging the fault and auxiliary planes leaves the moment unchanged, and bisecting P and T axes recovers both planes.
    swapped = core.double_couple(auxiliaries, planes, seismic_moment)         # [4] Plane+ <- Point-
    assert ((swapped - moment).svdvals() < 1e-12).all()
    found, found_auxiliaries = core.fault_planes(moment)                      # [4, 2] Plane+, [4, 2] Plane+
    rebuilt = core.double_couple(found, found_auxiliaries, seismic_moment)    # [4, 2] Plane+ <- Point-
    assert ((rebuilt - moment[:, None]).svdvals() < 1e-12).all()
    assert ((found.dot(planes[:, None]).abs().max(axis=-1) - 1.0).abs() < 1e-12).all()
    # P-wave motion equals the principal values along the principal axes and vanishes along the slip direction.
    axis_push, _ = core.radiation(moment[:, None], axes)                      # [4, 3] Scalar+
    assert ((axis_push - values).abs() < 1e-12).all()
    silent, _ = core.radiation(moment, auxiliaries.dual())                    # [4] Scalar+
    assert (silent.abs() < 1e-12).all()
    # Across the streamed rake sweep, the fault normal stays fixed, slip stays in the fault plane, and half a turn flips the moment.
    sweep_normals = stack([state[0] for state in states])                     # [n_frames] Point-
    sweep_slips = stack([state[1] for state in states])                       # [n_frames] Point-
    sweep_moments = stack([state[6] for state in states])                     # [n_frames] Plane+ <- Point-
    assert ((sweep_normals - sweep_normals[0]).norm() < 1e-12).all()
    assert (sweep_normals.dot(sweep_slips).abs() < 1e-12).all()
    half = n_frames // 2
    assert ((sweep_moments[half:] + sweep_moments[:half]).svdvals() < 1e-12).all()


if __name__ == "__main__":
    main()
