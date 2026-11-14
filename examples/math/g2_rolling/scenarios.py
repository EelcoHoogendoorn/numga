"""Split-octonion multiplication, kernel projection, and split-G2 derivations generate three rolling-ball scenes."""

from collections.abc import Iterator

from numga import stack
from examples.math.g2_rolling import core
from examples.math.g2_rolling.core import Point, Rotor, State, mv

attitude_angle = -2.4
n_samples = 120
n_steps = 41
n_hold_frames = 13
duration_ms = 40
sphere_radius = 3
ball_radius = 1
elevation = 35
azimuth = -25
size = 520
panel_size = 370
symmetry_amount = 0.12


# --- math -----------------------------------------------------------------------------
def two_roll_route() -> State:
    """Split-octonion multiplication supplies the intermediate waypoint (`[3] State+`) between two orthogonal endpoints."""
    start = core.configuration(mv.z.dual(), mv.rotor())                              # [] State+
    finish = core.configuration(mv.x.dual(), (mv.xy * (-attitude_angle / 2)).exp())  # [] State+
    waypoint = core.hemisphere_lift(core.split_product(start, finish).cast(State), mv.x)   # [] State+
    return stack([start, waypoint, finish])                                          # [3] State+


def three_roll_route() -> State:
    """Kernel projection and one product construct a closed-contact route (`[4] State+`) with a prescribed attitude change."""
    start = core.configuration(mv.z.dual(), mv.rotor())                              # [] State+
    finish = core.configuration(mv.z.dual(), (mv.xy * (-attitude_angle / 2)).exp())  # [] State+
    preferred_side = mv.x                                                            # [] Plane+
    first, second = core.three_roll_waypoints(start, finish, preferred_side.dual())  # [] State+, State+
    waypoints = core.hemisphere_lift(stack([first, second]), preferred_side)         # [2] State+
    return stack([start, *waypoints, finish])                                        # [4] State+


# --- plumbing -------------------------------------------------------------------------
def recorded_stream(stream: Iterator, history: list) -> Iterator:
    """Yields every element of `stream` in one pass while appending each element to `history`."""
    for item in stream:
        history.append(item)
        yield item


def main() -> None:
    from examples.util.animation import save_animation
    from examples.math.g2_rolling import render

    two_roll_stops = two_roll_route()                                                # [3] State+
    three_roll_stops = three_roll_route()                                            # [4] State+
    two_roll_placements: list[tuple[Point, Rotor]] = []
    three_roll_placements: list[tuple[Point, Rotor]] = []

    for stops, history, name in (
        (two_roll_stops, two_roll_placements, "g2_two_rolls"),
        (three_roll_stops, three_roll_placements, "g2_three_rolls"),
    ):
        frames = render.route_frames(
            core.contact_arcs(stops, n_samples),
            sphere_radius, ball_radius, elevation, azimuth, size,
            recorded_stream(core.route_motion(stops, n_steps, n_hold_frames), history),
        )
        save_animation(frames, name, duration_ms)

    # The third composition of this derivation vanishes, so its exponential truncates at degree two.
    generator = core.derivation(three_roll_stops[0], mv.x)                           # [] State+ <- State+
    symmetry = State + symmetry_amount * generator + symmetry_amount**2 / 2 * generator(generator)   # [] State+ <- State+
    transformed_stops = symmetry(three_roll_stops)                                   # [4] State+
    paired_placements: list[tuple[tuple[Point, Rotor], tuple[Point, Rotor]]] = []
    comparison = render.comparison_frames(
        core.contact_arcs(three_roll_stops, n_samples),
        core.contact_arcs(transformed_stops, n_samples),
        sphere_radius, ball_radius, elevation, azimuth, panel_size,
        recorded_stream(
            zip(
                core.route_motion(three_roll_stops, n_steps, n_hold_frames),
                core.route_motion(transformed_stops, n_steps, n_hold_frames),
            ),
            paired_placements,
        ),
    )
    save_animation(comparison, "g2_route_symmetry", duration_ms)

    # checks
    # Every stop lies on the split-octonion null cone, and consecutive stops multiply to zero.
    for stops in (two_roll_stops, three_roll_stops, transformed_stops):
        assert (core.pairing(stops, stops).abs() < 1e-10).all()
        assert (core.split_product(stops[:-1], stops[1:]).scalar_norm_squared() < 1e-10).all()

    third_power = generator(generator(generator))                                    # [] State+ <- State+
    assert (third_power.adjoint()(third_power).trace() < 1e-12).all()

    for placements in (
        two_roll_placements,
        three_roll_placements,
        [pair[0] for pair in paired_placements],
        [pair[1] for pair in paired_placements],
    ):
        contacts = stack([contact for contact, _ in placements])                     # [...] Point-
        attitudes = stack([attitude for _, attitude in placements])                  # [...] Rotor+
        orbit_turn = (1 + contacts[1:].dual() * contacts[:-1].dual()).normalized()   # [...] Rotor+
        ball_turn = orbit_turn * orbit_turn * orbit_turn * orbit_turn                # [...] Rotor+
        assert (((ball_turn * attitudes[:-1]) - attitudes[1:]).scalar_norm_squared() < 1e-10).all()


if __name__ == "__main__":
    main()
