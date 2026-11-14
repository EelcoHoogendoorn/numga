"""Unit-circle reflection in its own plane and circle-preserving rigid and conformal motions."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack

from examples.orientation.conformal_circle import core
from examples.orientation.conformal_circle.core import Point, Sphere, mv

# Frame duration in milliseconds and scenario sampling counts.
frame_duration_ms = 40
n_samples, n_meridians, n_parallels = 96, 16, 9
n_frames, n_radii, n_angles = 112, 40, 96
glide_amplitude = 0.9

# Ordered points on the horizontal unit circle (`+x`, `-y`, `-x`), cutting surfaces, and conformal reference elements.
circle_points = stack([mv.x.dual() + mv.e.dual(), -mv.y.dual() + mv.e.dual(),
                       -mv.x.dual() + mv.e.dual()])                             # [3] Point-
cutting_plane = mv.z                                                            # [] Sphere+
cutting_sphere = mv.w                                                           # [] Sphere+
origin = (mv.e.dual() - mv.w.dual()) / 2                                        # [] Point-
infinity = mv.e.dual() + mv.w.dual()                                            # [] Point-
horizon = mv.e + mv.w                                                           # [] Sphere+
reflector = mv.x                                                                # [] Sphere+
fixed_points = stack([mv.y.dual() + mv.e.dual(), -mv.y.dual() + mv.e.dual()])   # [2] Point-


# --- math -----------------------------------------------------------------------------
def reflected_circle_scene() -> tuple[Point, Point, Point, Sphere, Sphere]:
    """Sampling the horizontal unit circle, its cutting sphere, and their reflections in `z = 0` yields
    `[2, 3] Point-` markers, `[2, n_samples] Point-` curves, `[2, n_parallels, n_meridians] Point-`
    spheres, and `[2] Sphere+` planes and spheres."""
    mirror = cutting_plane                                                      # [] Sphere+
    angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    longitudes = np.linspace(0, 2 * np.pi, n_meridians, endpoint=False)
    latitudes = np.linspace(-np.pi / 2, np.pi / 2, n_parallels)

    turns = (mv.xy * (angles / 2)).exp()                                        # [n_samples] Rotor+
    curve = turns >> circle_points[0]                                           # [n_samples] Point-
    surface = core.sphere_samples(circle_points[0], mv.xy, mv.xz, longitudes, latitudes)  # [n_parallels, n_meridians] Point-
    markers = stack([circle_points, mirror >> circle_points])                   # [2, 3] Point-
    curves = stack([curve, mirror >> curve])                                    # [2, n_samples] Point-
    surfaces = stack([surface, mirror >> surface])                              # [2, n_parallels, n_meridians] Point-
    planes = stack([cutting_plane, mirror >> cutting_plane])                    # [2] Sphere+
    spheres = stack([cutting_sphere, mirror >> cutting_sphere])                 # [2] Sphere+
    return markers, curves, surfaces, planes, spheres


def disk_motion_states() -> Iterator[Point]:
    """Yields `[2, n_radii + 1, n_angles] Point-` disk grids across `n_frames` phases of rigid rotation
    and hyperbolic glide reflection."""
    phases = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)
    radii = np.linspace(0, 1, n_radii + 1)
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

    point_join, _ = core.circle_constructions(circle_points, cutting_plane, cutting_sphere)
    axis = (point_join.dual() | horizon).normalized()                           # [] Circle+
    grid = core.disk_samples(mv.x.dual(), origin, infinity, axis, radii, angles)  # [n_radii + 1, n_angles] Point-
    return core.circle_motion_states(point_join, horizon, reflector, grid, phases, glide_amplitude)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.orientation.conformal_circle import render

    markers, curves, surfaces, planes, spheres = reflected_circle_scene()
    save_figure(render.circle_figure(markers, curves, surfaces, planes, spheres), "orientation_conformal_circle")

    first_pair: list[Point] = []

    def recording_first() -> Iterator[Point]:
        for index, pair in enumerate(disk_motion_states()):
            if index == 0:
                first_pair.append(pair)
            yield pair

    save_animation(render.circle_motions_animation(fixed_points, recording_first()),
                   "orientation_circle_motions", frame_duration_ms)

    # checks
    point_join, surface_meet = core.circle_constructions(markers[0], planes[0], spheres[0])
    rebuilt_join, rebuilt_meet = core.circle_constructions(markers[1], planes[1], spheres[1])
    mirror = planes[0]                                                          # [] Sphere+
    basis_v = mv.basis()                                                        # [5] Sphere+
    trivectors = basis_v[:, None, None] ^ basis_v[None, :, None] ^ basis_v[None, None, :]  # [5, 5, 5] Trivector+
    antibivectors = (basis_v[:, None] ^ basis_v[None, :]).dual()                # [5, 5] Antibivector-
    assert ((trivectors & ((mirror >> point_join) - point_join)).abs() < 1e-12).all()
    assert (((mirror >> surface_meet) + surface_meet) & antibivectors).abs().max() < 1e-12
    assert ((trivectors & (rebuilt_join - (mirror >> point_join))).abs() < 1e-12).all()
    assert ((rebuilt_meet - (mirror >> surface_meet)) & antibivectors).abs().max() < 1e-12
    assert ((point_join & first_pair[0][:, -1]).antinorm().abs() < 1e-12).all()


if __name__ == "__main__":
    main()
