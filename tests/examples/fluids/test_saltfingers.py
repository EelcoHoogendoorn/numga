"""Salt fingers: gravity's pull on parcel mass alone, and one step of the fingering tank drawn."""

from examples.fluids.saltfingers import core, render, scenarios


def test_gravity_pulls_mass_alone():
    pull = -core.mv.ym                                                        # [] Drive+
    for part, expected in ((core.mass, -1.0), (core.volume, 0.0), (core.entropy, 0.0), (core.salt, 0.0)):
        assert (((core.mv.y & pull.commutator(part)) - expected).abs() < 1e-12).all()


def test_the_tank_steps_one_frame_and_draws():
    closed, opened, opening, vertices, temperature, salinity, frames = scenarios.fingers(
        n_along=3,
        n_across=3,
        n_flat=1,
    )
    render.draw_tank(opened, vertices, opening, scenarios.period, temperature, salinity)
    next(render.animate(opened, vertices, opening, scenarios.period, [next(frames)]))
