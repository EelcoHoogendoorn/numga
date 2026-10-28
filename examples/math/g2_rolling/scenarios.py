"""Scenes for octonionic rolling: a route of two rolls through the product of its ends, a route of three
rolls back to the same contact through the kernel of multiplication, and that route moved by a symmetry
of the product."""

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.math.g2_rolling import core
from examples.math.g2_rolling.core import Point, Rotor, State, mv

# The ball's turn in the horizontal plane between the ends of each route, in radians.
attitude_angle = -2.4
# The samples along each drawn arc, the steps of each roll, the frames held at each end, the duration of a
# frame; the radii of the sphere and the ball; the view's elevation and azimuth in degrees and the image
# sizes; and how far the symmetry moves the route.
n_samples = 120
n_steps = 36
n_hold_frames = 12
duration_ms = 45
sphere_radius = 3
ball_radius = 1
elevation = 35
azimuth = -25
size = 520
panel_size = 370
symmetry_amount = 0.12


# --- math -----------------------------------------------------------------------------
def two_roll_route() -> State:
    """A product supplies the intermediate configuration between two chosen endpoints."""
    start = core.encode(mv.z.dual(), mv.rotor())                       # State+
    finish = core.encode(mv.x.dual(), (mv.xy * (-attitude_angle / 2)).exp())   # State+
    waypoint = core.lift(core.multiply(start, finish).cast(State), mv.x)   # State+
    return stack([start, waypoint, finish])                            # [stops] State+


def three_roll_route() -> State:
    """A route back to the same contact point with a prescribed change of attitude."""
    start = core.encode(mv.z.dual(), mv.rotor())                       # State+
    finish = core.encode(mv.z.dual(), (mv.xy * (-attitude_angle / 2)).exp())   # State+
    # The plane through the centre on whose positive side the drawn contacts lie.
    preferred_side = mv.x                                              # Plane+
    first, second = core.three_rolls(start, finish, preferred_side.dual())   # State+, State+
    # Select the representatives in the preferred contact hemisphere.
    waypoints = core.lift(stack([first, second]), preferred_side)      # [waypoints] State+
    return stack([start, *waypoints, finish])                          # [stops] State+


def paths(stops: State, fractions: np.ndarray) -> tuple[Point, ...]:
    """The contact points along each leg of a route, sampled for drawing it."""
    states = core.roll(stops[:-1, None], stops[1:, None], fractions)   # [legs, samples] State+
    contacts, _ = core.decode(states)                                  # [legs, samples] Point-, Rotor+
    return tuple(contacts)


def rolling(stops: State, fractions: np.ndarray, n_hold_frames: int) -> Iterator[tuple[Point, Rotor]]:
    """The contact and attitude of each frame along the rolls of a route, with its ends held."""
    for _ in range(n_hold_frames):
        yield core.decode(stops[0])
    for first, second in zip(stops[:-1], stops[1:]):
        for fraction in fractions:
            yield core.decode(first * (1 - fraction) + second * fraction)
    for _ in range(n_hold_frames):
        yield core.decode(stops[-1])


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation
    from examples.math.g2_rolling import render

    fractions = np.linspace(0, 1, n_steps)
    samples = np.linspace(0, 1, n_samples)
    two_rolls = two_roll_route()                                       # [stops] State+
    three_rolls = three_roll_route()                                   # [stops] State+
    for stops, name in ((two_rolls, "g2_two_rolls"), (three_rolls, "g2_three_rolls")):
        images = render.frames(paths(stops, samples), rolling(stops, fractions, n_hold_frames),
                               sphere_radius, ball_radius, elevation, azimuth, size)
        save_animation(images, name, duration_ms)

    # The generator's third power vanishes, so its exponential ends at the second.
    generator = core.derivation(three_rolls[0], mv.x)                  # State+ <- State+
    symmetry = State + symmetry_amount * generator + symmetry_amount**2 / 2 * generator(generator)   # State+ <- State+
    transformed = symmetry(three_rolls)                                # [stops] State+
    images = render.comparison_frames(
        paths(three_rolls, samples), paths(transformed, samples),
        rolling(three_rolls, fractions, n_hold_frames), rolling(transformed, fractions, n_hold_frames),
        sphere_radius, ball_radius, elevation, azimuth, panel_size)
    save_animation(images, "g2_route_symmetry", duration_ms)

    # checks
    # Neighbouring stops of every route multiply to zero: each pair is one roll apart.
    for stops in (two_rolls, three_rolls, transformed):
        np.testing.assert_allclose(core.multiply(stops[:-1], stops[1:]).kernel, 0, atol=1e-10)


if __name__ == "__main__":
    main()
