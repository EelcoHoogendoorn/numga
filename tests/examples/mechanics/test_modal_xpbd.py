"""The scenes: a cantilever sags as the full truss does, a chain swings, a beam buckles."""

import numpy as np

from examples.mechanics.modal_xpbd import core, scenarios


def test_cantilever_static_sag_matches_full_truss():
    cells, stiffness = 3, 100.0
    shape = core.girder(cells, 1.0, 0.2, stiffness, 1.0, 2 * (2 * (cells + 1)) - 4)
    bodies, pins = scenarios.cantilever(shape, 1, np.array([2.0]), np.ones(1))
    initial = core.points(bodies, shape)
    dt, steps, acceleration = 0.01, 1200, -0.1
    gravity = (core.mv.y * acceleration).dual()
    for _ in range(steps):
        bodies = core.step(bodies, pins, dt, gravity)

    positions = shape.rest.dual().cast(core.Force).kernel
    difference = positions[shape.edges[:, 1]] - positions[shape.edges[:, 0]]
    lengths = np.linalg.norm(difference, axis=-1)
    directions = difference / lengths[:, None]
    incidence = np.eye(len(positions))[shape.edges[:, 1]] - np.eye(len(positions))[shape.edges[:, 0]]
    extension = (incidence[..., None] * directions[:, None, :]).reshape(len(shape.edges), -1)
    full_stiffness = extension.T @ ((stiffness / lengths)[:, None] * extension)
    forces = np.stack([np.zeros_like(shape.masses), shape.masses * acceleration], axis=-1).reshape(-1)
    reference = np.linalg.solve(full_stiffness[4:, 4:], forces[4:]).reshape(-1, 2)
    displacement = (core.points(bodies, shape) - initial).dual().cast(core.Force).kernel[0, 1]
    np.testing.assert_allclose(displacement[-1, 1], reference[-1, 1], rtol=0.02)
    assert np.linalg.norm(bodies.rate.kernel) < 1e-5
    np.testing.assert_allclose(core.points(bodies, shape).kernel[:, 0], initial.kernel[:, 0], atol=1e-11)


def test_hinged_chain_sustains_large_rotations_with_small_flex():
    shape = core.girder(4, 1.0, 0.2, 300.0, 1.0, 8)
    bodies, hinges = scenarios.hinged_chain(shape, 4, np.array([0.02]))
    initial = core.points(bodies, shape)
    gravity = (core.mv.y * -4).dual()
    dt, steps = 0.002, 800
    peak_gap = peak_flex = 0.0
    for _ in range(steps):
        bodies = core.step(bodies, hinges, dt, gravity)
        peak_gap = max(peak_gap, np.abs(core.coupling(bodies, hinges)[3].kernel).max())
        offsets = (shape.modes * bodies.amplitudes[..., None]).sum(axis=-2)
        peak_flex = max(peak_flex, offsets.dual().norm().kernel.max())
    kinetic = (bodies.rate & shape.inertia(bodies.rate)).sum(axis=-1) / 2
    kinetic = kinetic + bodies.rates.squared().sum(axis=(-1, -2)) / 2
    elastic = (bodies.amplitudes * bodies.frequencies).squared().sum(axis=(-1, -2)) / 2
    centres = bodies.motor >> core.mv.w.dual()
    potential = -((centres.dual() | gravity.dual()) * bodies.masses).sum(axis=-1)
    assert np.isfinite(core.points(bodies, shape).kernel).all()
    assert (kinetic + elastic + potential).kernel.max() < 1e-8
    assert peak_gap < 0.001
    assert 0.01 < peak_flex < 0.05
    assert np.abs((core.points(bodies, shape) - initial).kernel).max() > 2
    np.testing.assert_allclose((core.points(bodies, shape)[:, 0] - initial[:, 0]).kernel, 0, atol=1e-11)


def test_a_clamped_beam_buckles_past_its_euler_load():
    # Crushed from straight with no imperfection, the beam stays straight to round-off below its Euler
    # load and pops past it, the symmetry broken by rounding alone.
    beam = core.girder(scenarios.BEAM_CELLS, float(scenarios.BEAM_CELLS), scenarios.BEAM_HEIGHT, scenarios.BEAM_STIFFNESS, scenarios.DENSITY, scenarios.MODES)
    bodies, pins = scenarios.clamped_beam(beam, scenarios.BEAM_GIRDERS, scenarios.BEAM_DAMPING)
    crushing = scenarios.CRUSH * np.arange(scenarios.BEAM_FRAMES) / scenarios.BEAM_FRAMES
    midspans = [midspan for _, midspan in scenarios.crush(beam, bodies, pins, crushing, scenarios.BEAM_INTERVAL, scenarios.BEAM_SUBSTEPS)]
    deflection = np.array([abs(midspan.dual().cast(core.Force).kernel[0, 1]) for midspan in midspans])
    critical = scenarios.critical(scenarios.BEAM_HEIGHT, scenarios.BEAM_GIRDERS * scenarios.BEAM_CELLS)
    assert deflection[crushing < 0.8 * critical].max() < 1e-6
    assert deflection[crushing > 2 * critical].min() > 0.5
