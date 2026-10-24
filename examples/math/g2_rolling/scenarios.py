"""Scenes for octonionic rolling: a route of two rolls through the product of its ends, a route of three
rolls back to the same contact through the kernel of multiplication, and that route moved by a symmetry
of the product."""

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.math.g2_rolling import core
from examples.math.g2_rolling.core import Point, Rotor, State, mv, volume

# The ball's turn about the vertical between the ends of each route, in radians.
ATTITUDE_ANGLE = -2.4
# The samples along each drawn arc, the steps of each roll, the frames held at each end, the duration of a
# frame; the radii of the sphere and the ball; the view's elevation and azimuth in degrees and the image
# sizes; and how far the symmetry moves the route.
N_SAMPLES = 120
N_STEPS = 36
N_HOLD_FRAMES = 12
DURATION_MS = 45
SPHERE_RADIUS = 3
BALL_RADIUS = 1
ELEVATION = 35
AZIMUTH = -25
SIZE = 520
PANEL_SIZE = 370
SYMMETRY_AMOUNT = 0.12


# --- math -----------------------------------------------------------------------------
def two_roll_route() -> State:
    """A product supplies the intermediate configuration between two chosen endpoints."""
    start = core.encode(mv.z.dual(), mv.rotor())                       # State
    finish = core.encode(mv.x.dual(), (mv.xy * (-ATTITUDE_ANGLE / 2)).exp())   # State
    waypoint = core.lift(core.multiply(start, finish).cast(State), mv.x)  # State
    return stack([start, waypoint, finish])                            # [stops] State


def three_roll_route() -> State:
    """Return to the same contact point with a prescribed change of attitude."""
    start = core.encode(mv.z.dual(), mv.rotor())                       # State
    finish = core.encode(mv.z.dual(), (mv.xy * (-ATTITUDE_ANGLE / 2)).exp())   # State
    preference = mv.x * volume                                       # Bivector
    first, second = core.three_rolls(start, finish, preference)         # State, State
    # Select the representatives in the preferred contact hemisphere.
    return stack([start, core.lift(first, mv.x), core.lift(second, mv.x), finish])  # [stops] State


def paths(stops: State, fractions: np.ndarray) -> tuple[Point, ...]:
    """Sample each contact arc for the fixed route drawing."""
    states = core.roll(stops[:-1, None], stops[1:, None], fractions)    # [legs, samples] State
    contacts, _ = core.decode(states)                                 # [legs, samples] Point, Rotor
    return tuple(contacts)


def rolling(stops: State, fractions: np.ndarray, n_hold_frames: int) -> Iterator[tuple[Point, Rotor]]:
    """Yield the contact and attitude along each algebraically constructed roll."""
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

    fractions = np.linspace(0, 1, N_STEPS)
    samples = np.linspace(0, 1, N_SAMPLES)
    two_rolls = two_roll_route()                                      # [stops] State
    three_rolls = three_roll_route()                                  # [stops] State
    for stops, name in ((two_rolls, "g2_two_rolls"), (three_rolls, "g2_three_rolls")):
        images = render.frames(paths(stops, samples), rolling(stops, fractions, N_HOLD_FRAMES),
                               SPHERE_RADIUS, BALL_RADIUS, ELEVATION, AZIMUTH, SIZE)
        save_animation(images, name, DURATION_MS)

    # The generator's third power vanishes, so its exponential ends at the second.
    generator = core.derivation(three_rolls[0], mv.x)                  # State <- State
    symmetry = State + SYMMETRY_AMOUNT * generator + SYMMETRY_AMOUNT**2 / 2 * generator(generator)  # State <- State
    transformed = symmetry(three_rolls)                               # [stops] State
    images = render.comparison_frames(
        paths(three_rolls, samples), paths(transformed, samples),
        rolling(three_rolls, fractions, N_HOLD_FRAMES), rolling(transformed, fractions, N_HOLD_FRAMES),
        SPHERE_RADIUS, BALL_RADIUS, ELEVATION, AZIMUTH, PANEL_SIZE)
    save_animation(images, "g2_route_symmetry", DURATION_MS)

    # checks
    # Neighbouring stops of every route multiply to zero: each pair is one roll apart.
    for stops in (two_rolls, three_rolls, transformed):
        np.testing.assert_allclose(core.multiply(stops[:-1], stops[1:]).kernel, 0, atol=1e-10)


if __name__ == "__main__":
    main()
