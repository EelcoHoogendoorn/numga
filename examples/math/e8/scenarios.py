"""Assembles the 240-root E8 Coxeter orbits and renders their planar projections and four-plane rotation."""

from __future__ import annotations

from collections.abc import Iterator
from functools import reduce
from operator import mul

import numpy as np

from numga.extensor import stack

from examples.math.e8 import core
from examples.math.e8.core import Bivector, Rotor, Scalar, Vector


def root_system() -> tuple[Vector, Vector, Vector, Bivector, Scalar, Rotor, Vector]:
    """Eight 30-step Coxeter orbits (`[30, 8] Vector+`), four invariant plane frames and bivectors (`[4]`), winding rates, and rotor."""
    n_steps = core.coxeter_order
    simple_roots = core.simple_roots()                                        # [8] Vector+
    mirrors = simple_roots.normalized()                                       # [8] Vector+
    coxeter = reduce(mul, mirrors)                                            # [] Rotor+
    action = coxeter >> Vector                                                # [] Vector+ <- Vector+
    seeds = stack(core.reflection_seeds(simple_roots))                        # [8] Vector+
    roots = stack(core.orbit(action, seeds, n_steps))                         # [30, 8] Vector+
    plane_x, plane_y, planes, rates = core.invariant_planes(action, n_steps)  # [4] Vector+, Vector+, Bivector+, Scalar+
    return roots, plane_x, plane_y, planes, rates, coxeter, seeds


def main() -> None:
    """Saves the Coxeter-plane root projection and the four-plane rotation animation, then checks invariants."""
    from examples.math.e8 import render
    from examples.util.animation import save_animation, save_figure

    n_steps = core.coxeter_order
    n_substeps = 9
    n_frames = n_steps * n_substeps
    frame_duration_ms = 40
    roots, plane_x, plane_y, planes, rates, coxeter, seeds = root_system()
    save_figure(render.draw_roots(roots, plane_x[0], plane_y[0]), "e8_roots")

    rotor = core.step_rotor(planes, rates, n_steps, n_substeps)               # [] Rotor+
    advance = rotor >> Vector                                                 # [] Vector+ <- Vector+
    orbit_history: list[Vector] = []

    def orbit_stream() -> Iterator[Vector]:
        for frame_roots in core.orbit(advance, seeds, n_frames):
            orbit_history.append(frame_roots)
            yield frame_roots

    frames = render.planes_animation(roots, plane_x, plane_y, n_substeps, orbit_stream())
    save_animation(frames, "e8_planes", frame_duration_ms)

    # checks
    flat_roots = roots.reshape((-1,))                                         # [240] Vector+
    plane_gram = planes[:, None] | planes[None, :]                            # [4, 4] Scalar+
    assert len(flat_roots) == 240
    assert ((flat_roots.scalar_norm_squared() - 2).abs() < 1e-10).all()
    assert ((flat_roots[:, None] - flat_roots[None, :]).scalar_norm_squared() < 1e-6).sum() == 240
    assert (((coxeter >> roots[-1]) - seeds).scalar_norm_squared() < 1e-10).all()
    assert ((plane_gram + np.eye(4)).abs() < 1e-10).all()
    assert ((rates - np.array([1, 7, 11, 13])).abs() < 1e-10).all()
    assert ((stack(orbit_history[::n_substeps]) - roots).scalar_norm_squared() < 1e-10).all()


if __name__ == "__main__":
    main()
