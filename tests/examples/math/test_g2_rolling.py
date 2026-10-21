import numpy as np

from numga import stack
from examples.math.g2_rolling import core


def test_split_product_is_alternative_and_preserves_its_norm() -> None:
    rng = np.random.default_rng(8)
    first, second = core.mv(core.Octonion, rng.normal(size=(2, 8, 8)))
    product = core.multiply(first, second)
    np.testing.assert_allclose(
        core.pairing(product, product).kernel,
        (core.pairing(first, first) * core.pairing(second, second)).kernel,
        atol=1e-10,
    )
    repeated_left = core.multiply(core.multiply(first, first), second)
    repeated_right = core.multiply(first, core.multiply(first, second))
    np.testing.assert_allclose((repeated_left - repeated_right).kernel, 0, atol=1e-10)


def test_product_finds_the_two_roll_waypoint() -> None:
    start = core.encode(core.mv.z, core.mv.rotor())
    finish = core.encode(core.mv.x, (core.mv.xy * 0.6).exp())
    waypoint = core.normalize(core.multiply(start, finish).cast(core.State))

    np.testing.assert_allclose(core.pairing(start, finish).kernel, 0, atol=1e-12)
    assert np.linalg.norm(core.multiply(start, finish).kernel) > 0.1
    for first, second in ((start, waypoint), (waypoint, finish)):
        np.testing.assert_allclose(core.multiply(first, second).kernel, 0, atol=1e-12)
    contact, attitude = core.decode(finish)
    np.testing.assert_allclose((contact - core.mv.x).kernel, 0, atol=1e-12)
    np.testing.assert_allclose((attitude - (core.mv.xy * 0.6).exp()).kernel, 0, atol=1e-12)


def test_kernel_and_product_find_three_rolls_for_independent_endpoints() -> None:
    start = core.encode(core.mv.z, core.mv.rotor())
    finish = core.encode(core.mv.z, (core.mv.xy * 0.6).exp())
    first, second = core.three_rolls(start, finish, core.mv.x * core.volume)
    reachable = core.annihilator(start)

    assert abs(core.pairing(start, finish).kernel.item()) > 0.1
    assert np.linalg.matrix_rank(reachable.kernel, tol=1e-10) == 3
    np.testing.assert_allclose(core.multiply(start, reachable).kernel, 0, atol=1e-12)
    np.testing.assert_allclose((reachable(reachable) - reachable).kernel, 0, atol=1e-12)
    for left, right in ((start, first), (first, second), (second, finish)):
        np.testing.assert_allclose(core.multiply(left, right).kernel, 0, atol=1e-12)
    contacts, attitudes = core.decode(stack([start, first, second, finish]))
    np.testing.assert_allclose(contacts.scalar_norm_squared().kernel, 1, atol=1e-12)
    np.testing.assert_allclose(attitudes.scalar_norm_squared().kernel, 1, atol=1e-12)


def test_split_g2_deformation_preserves_products_and_physical_rolling() -> None:
    start = core.encode(core.mv.z, core.mv.rotor())
    finish = core.encode(core.mv.z, (core.mv.xy * 0.6).exp())
    first, second = core.three_rolls(start, finish, core.mv.x * core.volume)
    generator = core.derivation(start, core.mv.x)
    amount = 0.12
    symmetry = core.State + amount * generator + amount**2 / 2 * generator(generator)
    np.testing.assert_allclose(generator(generator(generator)).kernel, 0, atol=1e-12)

    rng = np.random.default_rng(17)
    left, right = core.mv(core.State, rng.normal(size=(2, 7, 7)))
    product = core.multiply(left, right)
    mapped_product = product.restrict[0] + symmetry(product.restrict.nonscalar())
    np.testing.assert_allclose(
        (mapped_product - core.multiply(symmetry(left), symmetry(right))).kernel,
        0, atol=1e-11,
    )

    # The same algebraic lines, before and after a G2 deformation, must roll physically.
    fractions = np.linspace(0.15, 0.85, 5)
    step = 1e-5
    for transform in (core.State, symmetry):
        waypoints = transform(stack([start, first, core.lift(second, core.mv.x), finish]))
        for left, right in zip(waypoints[:-1], waypoints[1:]):
            contact, attitude = core.decode(core.roll(left, right, fractions))
            contact_before, attitude_before = core.decode(core.roll(left, right, fractions - step))
            contact_after, attitude_after = core.decode(core.roll(left, right, fractions + step))
            contact_rate = (contact_after - contact_before) / (2 * step)
            attitude_rate = (attitude_after - attitude_before) / (2 * step)
            rotation_rate = 2 * attitude_rate.bivector_product(attitude.reverse())

            # The centre moves at radius four; rotation cancels its contact-point velocity.
            slip = 4 * contact_rate - rotation_rate.commutator(contact)
            # Rotation in the contact tangent plane would twist the ball about its contact.
            twist = rotation_rate.scalar_product(contact * core.volume)
            np.testing.assert_allclose(slip.kernel, 0, atol=1e-5)
            np.testing.assert_allclose(twist.kernel, 0, atol=1e-5)


def test_physical_route_lifts_commute_with_mirroring() -> None:
    mirror = (core.mv.x + core.mv.y + core.mv.z).normalized()
    start_contact = core.mv.z
    two_roll_contact = core.mv.x
    start_attitude = core.mv.rotor()
    finish_attitude = (core.mv.xy * 0.6).exp()
    hemisphere = core.mv.x

    start = core.encode(start_contact, start_attitude)
    two_roll_finish = core.encode(two_roll_contact, finish_attitude)
    three_roll_finish = core.encode(start_contact, finish_attitude)
    middle = core.lift(core.multiply(start, two_roll_finish).cast(core.State), hemisphere)
    first, second = core.three_rolls(start, three_roll_finish, hemisphere * core.volume)
    second = core.lift(second, hemisphere)

    mirror_start = core.encode(mirror >> start_contact, mirror >> start_attitude)
    mirror_two_finish = core.encode(mirror >> two_roll_contact, mirror >> finish_attitude)
    mirror_three_finish = core.encode(mirror >> start_contact, mirror >> finish_attitude)
    mirror_middle = core.lift(
        core.multiply(mirror_start, mirror_two_finish).cast(core.State), mirror >> hemisphere,
    )
    mirror_first, mirror_second = core.three_rolls(
        mirror_start, mirror_three_finish, (mirror >> hemisphere) * core.volume,
    )
    mirror_second = core.lift(mirror_second, mirror >> hemisphere)

    contact, attitude = core.decode(stack([middle, first, second]))
    mirror_contact, mirror_attitude = core.decode(stack([mirror_middle, mirror_first, mirror_second]))
    np.testing.assert_allclose((mirror_contact - (mirror >> contact)).kernel, 0, atol=1e-11)
    np.testing.assert_allclose((mirror_attitude - (mirror >> attitude)).kernel, 0, atol=1e-11)
