import numpy as np

from numga import stack
from examples.math.g2_rolling import core, render


def test_split_product_is_alternative_and_preserves_its_norm() -> None:
    rng = np.random.default_rng(8)
    first, second = core.mv(core.Octonion, rng.normal(size=(2, 4, 8)))
    product = core.split_product(first, second)
    assert ((core.pairing(product, product) - core.pairing(first, first) * core.pairing(second, second)).abs() < 1e-10).all()

    left_nested = core.split_product(core.split_product(first, first), second)
    right_nested = core.split_product(first, core.split_product(first, second))
    assert ((left_nested - right_nested).scalar_norm_squared() < 1e-10).all()


def test_product_finds_the_two_roll_waypoint() -> None:
    start = core.configuration(core.mv.z.dual(), core.mv.rotor())
    finish_attitude = (core.mv.xy * 0.6).exp()
    finish = core.configuration(core.mv.x.dual(), finish_attitude)
    waypoint = core.hemisphere_lift(core.split_product(start, finish).cast(core.State), core.mv.x)

    assert (core.pairing(start, finish).abs() < 1e-10).all()
    assert (core.split_product(start, finish).scalar_norm_squared() > 0.01).all()
    for left, right in ((start, waypoint), (waypoint, finish)):
        assert (core.split_product(left, right).scalar_norm_squared() < 1e-10).all()
    contact, attitude = core.contact_and_attitude(finish)
    assert ((contact - core.mv.x.dual()).scalar_norm_squared() < 1e-10).all()
    assert ((attitude - finish_attitude).scalar_norm_squared() < 1e-10).all()


def test_kernel_and_product_find_three_rolls_for_independent_endpoints() -> None:
    start = core.configuration(core.mv.z.dual(), core.mv.rotor())
    finish = core.configuration(core.mv.z.dual(), (core.mv.xy * 0.6).exp())
    first, second = core.three_roll_waypoints(start, finish, core.mv.x.dual())
    first_roll_space = core.annihilator(start)

    assert (core.pairing(start, finish).abs() > 0.1).all()
    assert ((first_roll_space.trace() - 3).abs() < 1e-10).all()
    idempotence_error = first_roll_space(first_roll_space) - first_roll_space
    assert (idempotence_error.adjoint()(idempotence_error).trace() < 1e-10).all()
    kernel_product = core.split_product(start, first_roll_space)
    assert (kernel_product.adjoint()(kernel_product).trace() < 1e-10).all()

    stops = stack([start, first, second, finish])
    assert (core.split_product(stops[:-1], stops[1:]).scalar_norm_squared() < 1e-10).all()
    contacts, attitudes = core.contact_and_attitude(stops)
    assert ((contacts.scalar_norm_squared() - 1).abs() < 1e-10).all()
    assert ((attitudes.scalar_norm_squared() - 1).abs() < 1e-10).all()


def test_split_g2_deformation_preserves_products_and_physical_rolling() -> None:
    start = core.configuration(core.mv.z.dual(), core.mv.rotor())
    finish = core.configuration(core.mv.z.dual(), (core.mv.xy * 0.6).exp())
    first, second = core.three_roll_waypoints(start, finish, core.mv.x.dual())
    generator = core.derivation(start, core.mv.x)
    amount = 0.12
    symmetry = core.State + amount * generator + amount**2 / 2 * generator(generator)
    third_power = generator(generator(generator))
    assert (third_power.adjoint()(third_power).trace() < 1e-10).all()

    rng = np.random.default_rng(17)
    left, right = core.mv(core.State, rng.normal(size=(2, 4, 7)))
    product = core.split_product(left, right)
    mapped_product = product.restrict[0] + symmetry(product.restrict.nonscalar())
    assert ((mapped_product - core.split_product(symmetry(left), symmetry(right))).scalar_norm_squared() < 1e-10).all()

    base_stops = stack([start, first, core.hemisphere_lift(second, core.mv.x), finish])
    for transform in (core.State, symmetry):
        stops = transform(base_stops)
        placements = list(core.route_motion(stops, 4, 1))
        contacts = stack([contact for contact, _ in placements])
        attitudes = stack([attitude for _, attitude in placements])
        orbit_turn = (1 + contacts[1:].dual() * contacts[:-1].dual()).normalized()
        ball_turn = orbit_turn * orbit_turn * orbit_turn * orbit_turn
        assert (((ball_turn * attitudes[:-1]) - attitudes[1:]).scalar_norm_squared() < 1e-10).all()


def test_physical_route_lifts_commute_with_mirroring() -> None:
    mirror = (core.mv.x + core.mv.y + core.mv.z).normalized()
    start_contact = core.mv.z.dual()
    two_roll_contact = core.mv.x.dual()
    start_attitude = core.mv.rotor()
    finish_attitude = (core.mv.xy * 0.6).exp()
    hemisphere = core.mv.x

    start = core.configuration(start_contact, start_attitude)
    two_roll_finish = core.configuration(two_roll_contact, finish_attitude)
    three_roll_finish = core.configuration(start_contact, finish_attitude)
    middle = core.hemisphere_lift(core.split_product(start, two_roll_finish).cast(core.State), hemisphere)
    first, second = core.three_roll_waypoints(start, three_roll_finish, hemisphere.dual())
    second = core.hemisphere_lift(second, hemisphere)

    mirror_start = core.configuration(mirror >> start_contact, mirror >> start_attitude)
    mirror_two_finish = core.configuration(mirror >> two_roll_contact, mirror >> finish_attitude)
    mirror_three_finish = core.configuration(mirror >> start_contact, mirror >> finish_attitude)
    mirror_middle = core.hemisphere_lift(
        core.split_product(mirror_start, mirror_two_finish).cast(core.State), mirror >> hemisphere,
    )
    mirror_first, mirror_second = core.three_roll_waypoints(
        mirror_start, mirror_three_finish, mirror >> hemisphere.dual(),
    )
    mirror_second = core.hemisphere_lift(mirror_second, mirror >> hemisphere)

    contact, attitude = core.contact_and_attitude(stack([middle, first, second]))
    mirror_contact, mirror_attitude = core.contact_and_attitude(stack([mirror_middle, mirror_first, mirror_second]))
    assert ((mirror_contact - (mirror >> contact)).scalar_norm_squared() < 1e-10).all()
    assert ((mirror_attitude - (mirror >> attitude)).scalar_norm_squared() < 1e-10).all()


def test_render_route_and_comparison_frames() -> None:
    start = core.configuration(core.mv.z.dual(), core.mv.rotor())
    finish = core.configuration(core.mv.x.dual(), (core.mv.xy * 0.4).exp())
    waypoint = core.hemisphere_lift(core.split_product(start, finish).cast(core.State), core.mv.x)
    stops = stack([start, waypoint, finish])
    arcs = core.contact_arcs(stops, 4)
    first_placement = core.contact_and_attitude(start)

    single_frame = next(render.route_frames(arcs, 3, 1, 35, -25, 24, [first_placement]))
    assert single_frame.shape == (24, 24, 3)
    assert single_frame.dtype == np.uint8

    comparison_frame = next(
        render.comparison_frames(arcs, arcs, 3, 1, 35, -25, 20, [(first_placement, first_placement)]),
    )
    assert comparison_frame.shape == (20, 40, 3)
    assert comparison_frame.dtype == np.uint8
