"""Newtonian gravity of soft Plummer masses, their tidal map, and first-order cluster deformation.

Each gravitating source is modelled as a Plummer sphere whose density stays finite inside its core
radius and falls off into an inverse-square pull at large distance. Summing over the sources gives
both the gravitational acceleration and the tidal map `Plane+ <- Point-`, which sends an intrinsic
displacement to the extrinsic plane reading the first-order change in gravity.

Multiplying each output plane of the tidal map by the plane reading its input step and taking the
trace `(Plane * tides(Point)).trace(1, 2)` forms the geometric derivative of gravity. Its bivector
curl vanishes by symmetry without invoking the metric, while its scalar divergence recovers minus
four pi times the Plummer mass density through Poisson's equation. Along a falling star cluster's
guiding centre, dualizing `tides(deformation).dual()` turns the plane reading into the tidal
acceleration of the linear deformation map `Point- <- Point-`, pushing an initial round spread
`cluster_radius**2 * Plane.dual()` forward to `deformation(start_spread(deformation.adjugate()))`
whose level curve `predicted_spread.solve(offsets) & offsets == 1` marks the deformed rim.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import NumpyContext
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                 # Scalar+
# Positions, displacements, velocities and accelerations are intrinsic antivectors; planes are extrinsic vectors.
Point = ga.gatype.antivector()                              # Point-
Plane = ga.gatype.vector()                                  # Plane+
Even = ga.gatype.even()                                     # Even+
Tidal = ga.gatype((Plane, Point))                           # Plane+ <- Point-
Deformation = ga.gatype((Point, Point))                     # Point- <- Point-
Spread = ga.gatype((Point, Plane))                          # Point- <- Plane+


def gravitational_acceleration(
    points: Point, centres: Point, masses: np.ndarray, core_radius: float
) -> Point:
    """Plummer spheres at `centres` pull each sample in `points` toward them (`[...] Point-`)."""
    source_offsets = centres - points[..., None]                             # [...] Point-
    softened_squares = source_offsets.scalar_norm_squared() + core_radius**2  # [...] Scalar+
    pull_strength = masses / (softened_squares * softened_squares.square_root())  # [...] Scalar+
    return (pull_strength * source_offsets).sum(axis=-1)                     # [...] Point-


def tidal_map(
    points: Point, centres: Point, masses: np.ndarray, core_radius: float
) -> Tidal:
    """Each Plummer sphere combines an isotropic squeeze with a radial stretch into the tidal map at `points` (`[...] Plane+ <- Point-`)."""
    source_offsets = centres - points[..., None]                             # [...] Point-
    softened_squares = source_offsets.scalar_norm_squared() + core_radius**2  # [...] Scalar+
    pull_strength = masses / (softened_squares * softened_squares.square_root())  # [...] Scalar+
    pull_slope = -1.5 * pull_strength / softened_squares                     # [...] Scalar+
    # Each radial plane pairs with a displacement to read its component toward that mass.
    source_planes = source_offsets.dual_inverse()                            # [...] Plane+
    squeeze_map = -pull_strength * Point.dual_inverse()                      # [...] Plane+ <- Point-
    stretch_map = -2 * pull_slope * source_planes * (source_planes & Point)  # [...] Plane+ <- Point-
    return (squeeze_map + stretch_map).sum(axis=-1)                          # [...] Plane+ <- Point-


def gravity_derivative(tides: Tidal) -> Even:
    """Tracing `Plane * tides(Point)` combines the scalar divergence and bivector curl of gravity (`[...] Even+`)."""
    # Tracing pairs each factor of Plane with the displacement argument of tides.
    return (Plane * tides(Point)).trace(1, 2)                                # [...] Even+


def sample_grid(
    half_width: float, half_height: float, n_columns: int, n_rows: int
) -> Point:
    """Cell centres span a `2 * half_width` by `2 * half_height` rectangle in the xy-plane (`[n_rows, n_columns] Point-`)."""
    x_offsets = (np.arange(n_columns) + 0.5) / n_columns * 2 * half_width - half_width
    y_offsets = (np.arange(n_rows) + 0.5) / n_rows * 2 * half_height - half_height
    return (mv.x * x_offsets[None, :] + mv.y * y_offsets[:, None]).dual()    # [n_rows, n_columns] Point-


def star_step(
    positions: Point,
    velocities: Point,
    centres: Point,
    masses: np.ndarray,
    core_radius: float,
    dt: float,
) -> tuple[Point, Point]:
    """Velocity Verlet advances star positions and velocities across one time step `dt` (`[...] Point-`, `[...] Point-`)."""
    start_acceleration = gravitational_acceleration(positions, centres, masses, core_radius)  # [...] Point-
    next_positions = positions + velocities * dt + start_acceleration * (dt**2 / 2)           # [...] Point-
    end_acceleration = gravitational_acceleration(next_positions, centres, masses, core_radius)  # [...] Point-
    next_velocities = velocities + (start_acceleration + end_acceleration) * (dt / 2)         # [...] Point-
    return next_positions, next_velocities


def deformation_step(
    centre: Point,
    velocity: Point,
    deformation: Deformation,
    deformation_rate: Deformation,
    centres: Point,
    masses: np.ndarray,
    core_radius: float,
    dt: float,
) -> tuple[Point, Point, Deformation, Deformation]:
    """Velocity Verlet advances the guiding centre and its tidal deformation across `dt` (`[] Point-`, `[] Point- <- Point-`)."""
    # Dualization turns the tidal map's plane reading into an intrinsic acceleration offset.
    start_acceleration = gravitational_acceleration(centre, centres, masses, core_radius)     # [] Point-
    start_tide = tidal_map(centre, centres, masses, core_radius)(deformation).dual()          # [] Point- <- Point-
    next_centre = centre + velocity * dt + start_acceleration * (dt**2 / 2)                   # [] Point-
    next_deformation = deformation + deformation_rate * dt + start_tide * (dt**2 / 2)         # [] Point- <- Point-
    end_acceleration = gravitational_acceleration(next_centre, centres, masses, core_radius)  # [] Point-
    end_tide = tidal_map(next_centre, centres, masses, core_radius)(next_deformation).dual()  # [] Point- <- Point-
    next_velocity = velocity + (start_acceleration + end_acceleration) * (dt / 2)             # [] Point-
    next_rate = deformation_rate + (start_tide + end_tide) * (dt / 2)                         # [] Point- <- Point-
    return next_centre, next_velocity, next_deformation, next_rate


def cluster_moment(positions: Point, centre: Point) -> Spread:
    """Star offsets around `centre` form the empirical second-moment map (`[...] Point- <- Plane+`)."""
    offsets = positions - centre[..., None]                                  # [...] Point-
    return (offsets * (Plane & offsets)).mean(axis=-1)                       # [...] Point- <- Plane+


def flyby_states(
    positions: Point,
    velocities: Point,
    guiding_centre: Point,
    guiding_velocity: Point,
    centres: Point,
    masses: np.ndarray,
    core_radius: float,
    cluster_radius: float,
    dt: float,
    n_substeps: int,
    n_frames: int,
) -> Iterator[tuple[Point, Point, Spread]]:
    """Yields `(positions, star_centre, predicted_spread)` across `n_frames` frames separated by `n_substeps` steps."""
    # Roundness enters through Plane.dual(), which maps each unit plane to the offset of equal length across it.
    start_spread = cluster_radius**2 * Plane.dual()                          # [] Point- <- Plane+
    deformation = Point                                                      # [] Point- <- Point-
    deformation_rate = 0 * Point                                             # [] Point- <- Point-
    for _ in range(n_frames):
        star_centre = positions.mean(axis=-1)                                # [] Point-
        predicted_spread = deformation(start_spread(deformation.adjugate()))  # [] Point- <- Plane+
        yield positions, star_centre, predicted_spread
        for _ in range(n_substeps):
            positions, velocities = star_step(
                positions, velocities, centres, masses, core_radius, dt
            )
            guiding_centre, guiding_velocity, deformation, deformation_rate = deformation_step(
                guiding_centre, guiding_velocity, deformation, deformation_rate, centres, masses, core_radius, dt
            )
