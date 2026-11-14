"""Coverage tests verify boosted soft-charge fields, Maxwell incidence relations, circling-charge
radiation, and rendering."""

import matplotlib.pyplot as plt
import numpy as np

from numga import stack

from examples.electromagnetism.moving_charge import core, render, scenarios


def test_maxwell_by_incidence_with_the_metric_in_the_vacuum():
    boost = (core.mv.xt * (scenarios.top_rapidity / 2)).exp()                    # [] Rotor+
    origin = 0 * core.time                                                       # [] Antivector-
    spacetime_axes = stack((core.mv.x, core.mv.y, core.mv.z, core.mv.t))         # [4] Vector+
    n_columns, n_rows = 6, 8
    events = core.event_grid(scenarios.half_width, scenarios.half_height, n_columns, n_rows)  # [n_rows, n_columns] Antivector-

    # Homogeneous Maxwell identity: the exterior derivative of the field gradient vanishes by incidence alone.
    field_gradients = core.field_gradient(
        np.array(scenarios.charge_strength), scenarios.core_radius, boost, origin, events
    )                                                                            # [n_rows, n_columns] Bivector+ <- Antivector-
    homogeneous_residual = (core.Vector ^ field_gradients(core.Antivector)).trace(1, 2)  # [n_rows, n_columns] Antivector+
    assert ((spacetime_axes ^ homogeneous_residual[..., None]).dual().abs() < 1e-11).all()

    # Lorenz gauge: the exterior derivative of the dualized potential gradient vanishes.
    potential_gradients = core.potential_gradient(
        np.array(scenarios.charge_strength), scenarios.core_radius, boost, origin, events
    )                                                                            # [n_rows, n_columns] Vector+ <- Antivector-
    lorenz_residual = (core.Vector ^ potential_gradients(core.Antivector).dual()).trace(1, 2)  # [n_rows, n_columns] Antiscalar-
    assert (lorenz_residual.dual().abs() < 1e-11).all()

    # Inhomogeneous Maxwell equation: the four-current equals the Plummer rest density convected along the boosted time step.
    rest_offsets = boost << events                                               # [n_rows, n_columns] Antivector-
    separations = rest_offsets - (core.mv.t & rest_offsets) * core.time          # [n_rows, n_columns] Antivector-
    squared_distance = scenarios.core_radius**2 + (separations | separations)    # [n_rows, n_columns] Scalar+
    plummer_density = (
        3 * scenarios.charge_strength * scenarios.core_radius**2 / (4 * np.pi) / (squared_distance * squared_distance * squared_distance.square_root())
    )                                                                            # [n_rows, n_columns] Scalar+
    current = core.four_current(field_gradients)                                 # [n_rows, n_columns] Antivector-
    plummer_current = plummer_density * (boost >> core.time)                     # [n_rows, n_columns] Antivector-
    assert ((spacetime_axes & (current - plummer_current)[..., None]).abs() < 1e-9).all()
    assert ((core.mv.t & current) > 0).all()

    # Exercise static figures and boost frame generator on the small grid without writing files.
    fields = core.field(potential_gradients)[None, ...]                          # [1, n_rows, n_columns] Bivector+
    fig_fields = render.fields_figure(events, fields)
    fig_densities = render.densities_figure(events, current[None, ...])
    plt.close(fig_fields)
    plt.close(fig_densities)
    frame = next(render.boost_frames([(events, fields[0], current)]))
    assert frame.shape == (n_rows, 2 * n_columns + render.panel_gap, 3)


def test_the_circling_charge_is_seen_along_light_rays():
    orbit_speed = float(scenarios.orbit_speeds[-1])
    n_grid = 6
    events = core.event_grid(scenarios.orbit_half_width, scenarios.orbit_half_width, n_grid, n_grid) + core.time * 3.0

    # Retarded emission events lie on the past light cone of each observation event.
    position, _, _ = core.worldline(
        scenarios.orbit_radius, orbit_speed, core.retarded_time(scenarios.orbit_radius, orbit_speed, events)
    )
    rays = events - position                                                     # [n_grid, n_grid] Antivector-
    assert ((rays | rays).abs() < 1e-10).all()

    # The Liénard–Wiechert potential gradient satisfies the Lorenz gauge.
    potentials = core.orbit_potential_gradient(
        scenarios.orbit_charge, scenarios.orbit_radius, orbit_speed, events
    )                                                                            # [n_grid, n_grid] Vector+ <- Antivector-
    lorenz_residual = (core.Vector ^ potentials(core.Antivector).dual()).trace(1, 2)  # [n_grid, n_grid] Antiscalar-
    assert (lorenz_residual.dual().abs() < 1e-11).all()

    # Slow and far away, the radial electric field matches Coulomb's inverse-square law.
    distance = 29.0
    far_event = (scenarios.orbit_radius + distance) * core.mv.x.dual()           # [] Antivector-
    outward_field = core.field(
        core.orbit_potential_gradient(scenarios.orbit_charge, scenarios.orbit_radius, 1e-3, far_event)
    ) & (core.mv.x.dual() & core.time)                                           # [] Scalar+
    coulomb_field = scenarios.orbit_charge / (4 * np.pi * distance**2)
    assert ((outward_field - coulomb_field).abs() < 1e-2 * coulomb_field).all()

    # Exercise GPU radiation renderer for 1 frame and static figure on a tiny viewport without saving files.
    start = scenarios.orbit_radius * core.mv.x.dual()                            # [] Antivector-
    n_test_pixels = 32
    fig_radiation = render.radiation_figure(
        core.orbit_field_source,
        scenarios.orbit_charge,
        scenarios.orbit_radius,
        scenarios.orbit_speeds[:1],
        start,
        scenarios.orbit_half_width,
        n_test_pixels,
    )
    plt.close(fig_radiation)
    wave_frame = next(
        render.wave_frames(
            core.orbit_field_source,
            scenarios.orbit_charge,
            scenarios.orbit_radius,
            orbit_speed,
            scenarios.orbit_half_width,
            n_test_pixels,
            [(0.0, start)],
        )
    )
    assert wave_frame.shape == (n_test_pixels, n_test_pixels, 3)
