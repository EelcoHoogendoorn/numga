"""A cluster of stars falling past a soft mass and stretched into a stream by its tides."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.mechanics.tides import core

MASS = 1.0
MASS_CORE = 0.3
STARS = 300
CLUSTER_RADIUS = 0.25


# --- math -----------------------------------------------------------------------------
def setting() -> tuple[core.Masses, core.Stars, core.Shape]:
    """The mass, the cluster as a disc of stars in the plane z = 0, and its shape."""
    masses = core.Masses(core.mv.vector([[0.0, 0.0, 0.0]]), np.array([MASS]), MASS_CORE)
    seed, start, start_velocity = 1, [-3.0, 0.5, 0.0], [0.5, 0.0, 0.0]
    rng = np.random.default_rng(seed)
    # Uniform over the disc: a radius growing as the square root, turned by a rotor to a random angle.
    radii = CLUSTER_RADIUS * np.sqrt(rng.uniform(size=STARS))
    turns = (core.mv.xy * (-np.pi * rng.uniform(size=STARS))).exp()           # [stars] Rotor
    offsets = turns >> (core.mv.x * radii)                                   # [stars] Vector
    centre, velocity = core.mv.vector(start), core.mv.vector(start_velocity)  # [] Vector each
    stars = core.Stars(centre + offsets, velocity.broadcast_to(offsets.shape))
    shape = core.Shape(centre, velocity, 1 * core.Vector, 0 * core.Vector)
    return masses, stars, shape


def flyby(frames: int) -> Iterator[tuple[core.Stars, core.Shape]]:
    """The stars, and the cluster's shape as the tidal map deforms it, frame by frame."""
    dt, substeps = 0.005, 10
    masses, stars, shape = setting()
    for _ in range(frames):
        yield stars, shape
        for _ in range(substeps):
            stars, shape = core.fall(masses, stars, dt), core.deform(masses, shape, dt)


def outline(stars: core.Stars, shape: core.Shape) -> tuple[core.Vector, core.Form]:
    """The stars' centre of mass and the rim the deformation predicts around it. The shape is first
    order in the cluster's size; the drift of its centre of mass off the orbit is second order."""
    return stars.centre(), core.rim(shape, CLUSTER_RADIUS)


def field() -> tuple[core.Vector, core.Even]:
    """The derivative of gravity on the plane z = 0."""
    half_width, half_height, columns, rows = 3.6, 2.4, 360, 240
    masses, _, _ = setting()
    points = core.grid(half_width, half_height, columns, rows)              # [rows, columns] Vector
    return points, core.derivative(core.tidal(masses, points))


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation
    from examples.mechanics.tides import render

    # The flyby ends while the stream is still in the picture.
    frames, frame_ms = 120, 50
    points, derivative = field()
    states = list(flyby(frames))
    drawn = [(stars, *outline(stars, shape)) for stars, shape in states]
    save_animation(render.animate(points, derivative, drawn), "tides_flyby", frame_ms)

    # --- checks
    # The derivative of gravity is minus four pi times the Plummer density, and has no curl.
    masses, _, _ = setting()
    towards = masses.centres - points[..., None]
    softened = (towards | towards) + MASS_CORE**2
    density = (3 * masses.masses * MASS_CORE**2 / (4 * np.pi) / (softened * softened * softened.square_root())).sum(axis=-1)
    np.testing.assert_allclose((derivative + 4 * np.pi * density).kernel, 0.0, atol=1e-10)
    # Through the whole flyby the stars spread in the plane as the deformation predicts: a uniform
    # disc's moment is a quarter of its radius squared. Measured against the largest spread, the
    # sampling of the disc costs about a tenth, and the passage through the mass's core a quarter.
    directions = (core.mv.xy * (-np.pi * np.arange(8) / 8)).exp() >> core.mv.x   # [directions] Vector
    for stars, shape in states:
        measured = (directions | stars.moment()(directions)).to_array()      # [directions]
        predicted = (directions | (core.spread(shape, CLUSTER_RADIUS) / 4)(directions)).to_array()
        np.testing.assert_allclose(measured, predicted, rtol=0, atol=0.3 * predicted.max())


if __name__ == "__main__":
    main()
