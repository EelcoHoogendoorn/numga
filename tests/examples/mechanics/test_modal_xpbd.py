"""The scenes: a cantilever sags as the full truss does, a chain swings, a beam buckles."""

from dataclasses import replace
from functools import partial

import numpy as np
import pytest

from numga import stack
from numga.algebras import PGA2D
from examples import instantiate
from examples.mechanics.modal_xpbd import scenarios

core = instantiate("examples.mechanics.modal_xpbd.core", PGA2D)


def coordinates(points: core.Point) -> np.ndarray:
    """Coordinates `[..., 2]` of points of unit weight, or of displacements: their pairings with the planes x = 0 and y = 0."""
    return (stack((core.mv.x, core.mv.y)) & points.batch()[..., None]).to_array()


def test_cantilever_static_sag_matches_full_truss():
    cells, stiffness = 3, 100.0
    shape = core.girder(cells, 1.0, 0.2, stiffness, 1.0, 2 * (2 * (cells + 1)) - 4)
    bodies, constraints = scenarios.cantilever(core, shape, np.array([2.0]), np.ones((1, 1)))
    initial = core.points(bodies, shape)
    # Implicit compliance keeps large steps stable; the over-damped girder settles within them.
    dt, steps, acceleration = 0.2, 60, -0.1
    gravity = core.mv.direction([0, acceleration])
    for _ in range(steps):
        bodies = core.step(bodies, constraints, dt, gravity)

    positions = coordinates(shape.rest)
    difference = positions[shape.edges[:, 1]] - positions[shape.edges[:, 0]]
    lengths = np.linalg.norm(difference, axis=-1)
    directions = difference / lengths[:, None]
    incidence = np.eye(len(positions))[shape.edges[:, 1]] - np.eye(len(positions))[shape.edges[:, 0]]
    extension = (incidence[..., None] * directions[:, None, :]).reshape(len(shape.edges), -1)
    full_stiffness = extension.T @ ((stiffness / lengths)[:, None] * extension)
    masses = shape.masses.kernel[:, 0]
    forces = np.stack([np.zeros_like(masses), masses * acceleration], axis=-1).reshape(-1)
    reference = np.linalg.solve(full_stiffness[4:, 4:], forces[4:]).reshape(-1, 2)
    displacement = coordinates(core.points(bodies, shape) - initial)[0, 1]
    np.testing.assert_allclose(displacement[-1, 1], reference[-1, 1], rtol=0.02)
    assert np.linalg.norm(bodies.rate.kernel) < 1e-5
    np.testing.assert_allclose(core.points(bodies, shape).kernel[:, 0], initial.kernel[:, 0], atol=1e-11)


def test_hinged_chain_sustains_large_rotations_with_small_flex():
    shape = core.girder(4, 1.0, 0.2, 300.0, 1.0, 8)
    # Exactly rigid at the pin, then four times softer per link.
    bodies, hinges = scenarios.hinged_chain(core, shape, np.array([0.02]), np.array([[0.0, 1.0, 4.0, 16.0]]))
    initial = core.points(bodies, shape)
    gravity = core.mv.direction([0, -4])
    dt, steps = 0.008, 200
    peak_gap, peak_flex = 0.0, np.zeros(5)
    for _ in range(steps):
        bodies = core.step(bodies, hinges, dt, gravity)
        peak_gap = max(peak_gap, np.abs(core.coupling(bodies, hinges)[3].kernel).max())
        offsets = (shape.modes[:, None] * bodies.amplitudes.batch()).sum(axis=-2)
        peak_flex = np.maximum(peak_flex, (offsets.dual_inverse() & offsets).square_root().kernel[0, :, :, 0].max(axis=-1))
    kinetic = (bodies.rate & shape.inertia(bodies.rate)).sites.sum() / 2
    kinetic = kinetic + bodies.rates.squared().sites.sum().sum(axis=-1) / 2
    # A mode stores its amplitude squared over twice its compliance; the rigid girders' modes stay at rest and store
    # nothing.
    compliance, amplitudes = bodies.compliance.kernel, bodies.amplitudes.kernel
    elastic = np.divide(amplitudes**2, 2 * compliance, out=np.zeros_like(amplitudes), where=compliance > 0).sum()
    centres = bodies.motor >> core.mv.w.dual()
    potential = -((gravity.dual_inverse() & centres) * bodies.masses).sites.sum()
    assert np.isfinite(core.points(bodies, shape).kernel).all()
    assert (kinetic + potential).kernel.max() + elastic < 1e-8
    assert peak_gap < 0.001
    # The fixed girder and the rigid link do not flex at all; the flexible links flex more the softer they are, the
    # softest by a small fraction of its length.
    assert (peak_flex[:2] == 0).all()
    assert (np.diff(peak_flex[1:]) > 0).all()
    assert 0.05 < peak_flex[-1] < 0.2
    assert np.abs((core.points(bodies, shape) - initial).kernel).max() > 2
    np.testing.assert_allclose((core.points(bodies, shape)[:, 0] - initial[:, 0]).kernel, 0, atol=1e-11)


def test_a_clamped_beam_buckles_past_its_euler_load():
    beam = core.girder(scenarios.N_BEAM_CELLS, float(scenarios.N_BEAM_CELLS), scenarios.BEAM_HEIGHT, scenarios.BEAM_STIFFNESS, scenarios.DENSITY, scenarios.N_MODES)
    bodies, constraints = scenarios.clamped_beam(core, beam, scenarios.N_BEAM_GIRDERS, scenarios.BEAM_DAMPING)
    # The two middle girders lifted by a hair: an imperfection to buckle from, so that the buckle grows within
    # a few steps rather than from round-off.
    frames, substeps, dt, imperfection = 24, 4, 0.02, 1e-6
    lift = np.zeros(scenarios.N_BEAM_GIRDERS + 2)
    lift[scenarios.N_BEAM_GIRDERS // 2:scenarios.N_BEAM_GIRDERS // 2 + 2] = imperfection
    bodies = replace(bodies, motor=(bodies.motor * (core.mv.yw * (lift / 2)).exp().field()).cast(core.Motor))
    displacements = scenarios.END_DISPLACEMENT * np.arange(frames) / frames
    midspans = [midspan for _, midspan in scenarios.compress(core, beam, bodies, constraints, displacements, dt, substeps)]
    deflection = np.array([abs(coordinates(midspan)[0, 1]) for midspan in midspans])
    critical = scenarios.critical(scenarios.BEAM_HEIGHT, scenarios.N_BEAM_GIRDERS * scenarios.N_BEAM_CELLS)
    assert deflection[displacements < 0.8 * critical].max() < 1e-6
    assert deflection[displacements > 2 * critical].min() > 0.5


def test_the_derived_step_follows_the_sparse_step():
    """Couplings found by differentiating the gaps step the chain as the hand-built sparse couplings do."""
    pytest.importorskip("jax")
    import jax
    from examples.mechanics.modal_xpbd import derived

    dt, steps = 0.002, 100
    shape, jax_shape = core.girder(4, 1.0, 0.2, 300.0, 1.0, 8), derived.core.girder(4, 1.0, 0.2, 300.0, 1.0, 8)
    flexibility = np.array([[0.0, 1.0, 4.0, 16.0]])
    bodies, hinges = scenarios.hinged_chain(core, shape, np.array([0.02]), flexibility)
    jax_bodies, jax_hinges = scenarios.hinged_chain(derived.core, jax_shape, np.array([0.02]), flexibility)
    gravity, jax_gravity = core.mv.direction([0, -4]), derived.mv.direction([0, -4])
    jax_step = jax.jit(partial(derived.step, constraints=jax_hinges, dt=dt, gravity=jax_gravity))
    for _ in range(steps):
        bodies, jax_bodies = core.step(bodies, hinges, dt, gravity), jax_step(jax_bodies)
    # Each girder's modes are found again on JAX, and a mode's sign is arbitrary: its points are not.
    np.testing.assert_allclose(np.asarray(derived.core.points(jax_bodies, jax_shape).kernel), core.points(bodies, shape).kernel, atol=1e-10)
