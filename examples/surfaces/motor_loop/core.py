"""Loop subdivision refines a triangulated surface whose vertices carry rigid placements of frames rather than positions alone. A Loop step averages vertex values across the topology with fixed stencil weights determined solely by connectivity. When neighbouring frames twist relative to one another, averaging their positions cuts straight across the chord of the turn and wrings the surface thin, whereas averaging their rigid placements and normalizing back to a rigid motion interpolates along the arc between them and preserves the surface width.

Each placement is a motor, and each Loop step is a sparse linear map from coarse vertex sites to fine vertex sites that acts directly on `step * motors`. Because a weighted sum of motors leaves the group, normalizing after each step via `(step * motors).normalized()` returns every averaged element to a unit motor before the next refinement level, while sandwiching the origin `motors >> origin` traces out the surface.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import PGA3D as ga
from numga.topology import SubdividedTopology, Topology

ctx = NumpyContext(ga)
mv = ctx.multivector
Line = ga.gatype.bivector()                                                 # Bivector+
Point = ga.gatype.antivector()                                              # Antivector-
Motor = ga.gatype.rotor()                                                   # Rotor+
origin = mv.w.dual()                                                        # Point-


def subdivided_motors(topology: SubdividedTopology, motors: Motor) -> Motor:
    """Repeated Loop refinement averages `motors` across each level of `topology` and normalizes every sum back to a unit motor (`[...] Motor+[V_refined]`)."""
    for step in topology.steps:
        motors = (step * motors).normalized()                               # [...] Motor+[V_finer]
    return motors


def frame_markers(axis_length: float) -> Point:
    """Four reference points mark a frame's origin followed by the tips of its three axes at `axis_length` (`[4] Point-`)."""
    return stack([
        origin,
        mv.point([axis_length, 0.0, 0.0]),
        mv.point([0.0, axis_length, 0.0]),
        mv.point([0.0, 0.0, axis_length]),
    ])                                                                      # [4] Point-


def subdivided_surfaces(topology: SubdividedTopology, motors: Motor) -> Point:
    """Point subdivision refines the vertex origins directly while motor subdivision refines and normalizes `motors` at each level (`[..., 2] Point-[V_refined]`)."""
    point_surface = topology.subdivision * (motors >> origin)               # [...] Point-[V_refined]
    motor_surface = subdivided_motors(topology, motors) >> origin           # [...] Point-[V_refined]
    return stack([point_surface, motor_surface], axis=-1)                   # [..., 2] Point-[V_refined]


def ribbon_rest_motors(z_coords: np.ndarray, x_coords: np.ndarray) -> Motor:
    """Translations along `z_coords` and across `x_coords` place flat vertex frames across a `[V]` ribbon grid (`[V] Motor+`)."""
    return (mv.zw * (z_coords / 2)).exp() * (mv.xw * (x_coords / 2)).exp()  # [V] Motor+


def ribbon_states(topology: SubdividedTopology, markers: Point, motors: Motor) -> Iterator[tuple[Point, Point]]:
    """Yields `(control_frames, surfaces)` (`[4] Point-[V]`, `[2] Point-[V_refined]`) for each frame of `motors`."""
    for frame_motors in motors:
        yield frame_motors[None] >> markers, subdivided_surfaces(topology, frame_motors)


def winding_states(topology: SubdividedTopology, markers: Point, rest_motors: Motor,
                   z_coords: np.ndarray, length: float, twist_angle: float, n_frames: int) -> Iterator[tuple[Point, Point]]:
    """Yields `(control_frames, surfaces)` across `n_frames` as the ribbon winds up to `twist_angle` and unwinds back to rest."""
    twist_angles = twist_angle * (1 - np.cos(np.linspace(0, 2 * np.pi, n_frames, endpoint=False))) / 2  # [n_frames]
    twist_profile = ((z_coords + length / 2) / length) ** 2                 # [V]
    twist_turns = (mv.xy * (-twist_angles[:, None] * twist_profile / 2)).exp()  # [n_frames, V] Motor+
    motors = (twist_turns * rest_motors).field()                            # [n_frames] Motor+[V]
    yield from ribbon_states(topology, markers, motors)


def turning_states(topology: SubdividedTopology, markers: Point, rest_motors: Motor,
                   axes: Line, phases: np.ndarray, turn_amplitude: float, n_frames: int) -> Iterator[tuple[Point, Point]]:
    """Yields `(control_frames, surfaces)` across `n_frames` as vertex frames oscillate by `turn_amplitude` around `axes` at `phases`."""
    angles = turn_amplitude * np.sin(np.linspace(0, 2 * np.pi, n_frames, endpoint=False)[:, None] + phases)  # [n_frames, V]
    local_turns = (axes * (-angles / 2)).exp()                              # [n_frames, V] Motor+
    motors = (rest_motors * local_turns).field()                            # [n_frames] Motor+[V]
    yield from ribbon_states(topology, markers, motors)
