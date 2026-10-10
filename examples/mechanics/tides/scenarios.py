"""A cluster of stars falling past a soft mass and stretched into a stream by its tides."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.mechanics.tides import core

MASS = 1.0
MASS_CORE = 0.3
STARS = 300
CLUSTER_RADIUS = 0.25
SEED = 1
START = [-3.0, 0.5, 0.0]
START_VELOCITY = [0.5, 0.0, 0.0]
DT = 0.005
SUBSTEPS = 10
# The field's half extents and its sample counts in the plane z = 0.
HALF_WIDTH, HALF_HEIGHT = 3.6, 2.4
COLUMNS, ROWS = 360, 240
# The flyby ends while the stream is still in the picture.
FRAMES = 120
FRAME_MS = 50


# --- math -----------------------------------------------------------------------------
def setting() -> tuple[core.Masses, core.Stars, core.Shape]:
    """The mass, the cluster as a disc of stars in the plane z = 0, and its shape."""
    masses = core.Masses(core.mv.vector([[0.0, 0.0, 0.0]]), np.array([MASS]), MASS_CORE)
    rng = np.random.default_rng(SEED)
    # Uniform over the disc: a radius growing as the square root, turned by a rotor to a random angle.
    radii = CLUSTER_RADIUS * np.sqrt(rng.uniform(size=STARS))
    turns = (core.mv.xy * (-np.pi * rng.uniform(size=STARS))).exp()           # [stars] Rotor
    offsets = turns >> (core.mv.x * radii)                                   # [stars] Vector
    centre, velocity = core.mv.vector(START), core.mv.vector(START_VELOCITY)  # [] Vector each
    stars = core.Stars(centre + offsets, velocity.broadcast_to(offsets.shape))
    shape = core.Shape(centre, velocity, core.Vector, 0 * core.Vector)
    return masses, stars, shape


def flyby(frames: int) -> Iterator[tuple[core.Stars, core.Shape]]:
    """The stars, and the cluster's shape as the tidal map deforms it, frame by frame."""
    masses, stars, shape = setting()
    for _ in range(frames):
        yield stars, shape
        for _ in range(SUBSTEPS):
            stars, shape = stars.fall(masses, DT), shape.deform(masses, DT)


def outline(stars: core.Stars, shape: core.Shape) -> tuple[core.Vector, core.Form]:
    """The stars' centre of mass and the rim the deformation predicts around it. The shape is first
    order in the cluster's size; the drift of its centre of mass off the orbit is second order."""
    return stars.centre(), shape.rim(CLUSTER_RADIUS)


def field() -> tuple[core.Vector, core.Even, core.Scalar]:
    """The derivative of gravity on the plane z = 0, and the density it gives by Poisson's equation."""
    masses, _, _ = setting()
    points = core.grid(HALF_WIDTH, HALF_HEIGHT, COLUMNS, ROWS)              # [rows, columns] Vector
    derivative = core.derivative(masses.tidal(points))                      # [rows, columns] Even
    return points, derivative, -derivative.select[0] / (4 * np.pi)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation
    from examples.mechanics.tides import render

    points, derivative, density = field()
    states = list(flyby(FRAMES))
    drawn = [(stars, *outline(stars, shape)) for stars, shape in states]
    save_animation(render.animate(points, density, drawn), "tides_flyby", FRAME_MS)

    # --- checks
    # The derivative of gravity is minus four pi times the Plummer density, and has no curl.
    masses, _, _ = setting()
    towards = masses.centres - points[..., None]
    softened = (towards | towards) + MASS_CORE**2
    plummer = (3 * masses.masses * MASS_CORE**2 / (4 * np.pi) / (softened * softened * softened.square_root())).sum(axis=-1)
    np.testing.assert_allclose((derivative + 4 * np.pi * plummer).kernel, 0.0, atol=1e-10)
    # Through the whole flyby the stars spread in the plane as the deformation predicts: a uniform
    # disc's moment is a quarter of its radius squared. Measured against the largest spread, the
    # sampling of the disc costs about a tenth, and the passage through the mass's core a quarter.
    directions = (core.mv.xy * (-np.pi * np.arange(8) / 8)).exp() >> core.mv.x   # [directions] Vector
    for stars, shape in states:
        measured = (directions | stars.moment()(directions)).to_array()      # [directions]
        predicted = (directions | (shape.spread(CLUSTER_RADIUS) / 4)(directions)).to_array()
        np.testing.assert_allclose(measured, predicted, rtol=0, atol=0.3 * predicted.max())


if __name__ == "__main__":
    main()
