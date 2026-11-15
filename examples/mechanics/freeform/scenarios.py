"""A brief pull on one ear of an elastic Stanford bunny, held at its base, then released."""

from __future__ import annotations

from collections.abc import Iterator

import jax
import numpy as np

from numga import stack
from numga.backend.jax import JaxContext
from examples import MODEL_DIR
from examples.mechanics.freeform import core, geometry, weights
from examples.mechanics.freeform.core import ga, mv, Scalar, Antivector, Vector
from examples.mechanics.freeform.numerics import minimized_state

# The bunny's height, its volume samples and kernel centers, the kernels' radius, and the weight modes.
surface_path = MODEL_DIR / "bunny.obj"
height = 2
n_quadrature = 1600
n_centers = 96
radius = 0.3
n_modes = 9
# The material: its Lamé parameters and density.
lame_lambda = 20
lame_mu = 10
density = 1
# The compliant support at the base: its depth and stiffness.
clamp_depth = 0.12
clamp_stiffness = 500
# The pull near one ear's tip: where, how wide, the total force, and how long it lasts.
pull_center = mv.vector([-0.25, 0.34, 1.92]).dual()          # Antivector-
pull_width = 0.23
pull_force = (mv.x + 0.7 * mv.y) * 0.6                       # Vector+
pull_duration = 0.4
# The time step, the steps and the steps per frame drawn, and the minimizer's iterations and tolerance per step.
time_step = 0.02
n_steps = 200
n_steps_per_frame = 2
n_iterations = 30
tolerance = 1e-7
frame_duration_ms = round(time_step * n_steps_per_frame * 1000)


# --- math -----------------------------------------------------------------------------
def bunny_motion() -> tuple[Iterator[Antivector], geometry.Topology]:
    """Yields the triangulated surface vertices across frame steps from rest alongside the surface `Topology`."""
    topology, vertices, samples, sample_volumes, centers = geometry.load_solid(
        surface_path, height, n_quadrature, n_centers
    )
    V, E, F = topology.n_elements
    evaluation, gradient = weights.kernels(samples, centers, radius)
    _, modes = weights.skinning_modes(evaluation, gradient, sample_volumes, lame_lambda, lame_mu, n_modes)
    # Four control points around the bunny form the corners of an enclosing tetrahedron.
    corners = mv.vector([[-1, -1, 0], [1, -1, 2], [-1, 1, 2], [1, 1, 0]]).dual()   # [4] Antivector-
    n_controls = len(corners) * n_modes
    sampling, gradients = core.deformation_basis(samples, centers, radius, modes, corners)
    surface_sampling, _ = core.deformation_basis(vertices, centers, radius, modes, corners)
    mass = core.control_mass(sampling, sample_volumes * density)                     # (Vector+ <- Antivector-)[n_controls <- n_controls]

    # A compliant support holds the base while a localized profile pulls near one ear tip.
    clamp = (-(mv.z & samples).squared() / clamp_depth**2).exp() * clamp_stiffness   # [n_quadrature] Scalar+
    profile = (-(samples - pull_center).scalar_norm_squared() / pull_width**2).exp() # [n_quadrature] Scalar+
    # Normalizing over the solid distributes the stated total force across the profile.
    load = profile * pull_force / (profile * sample_volumes).sum(axis=0)             # [n_quadrature] Vector+
    pulse_strengths = np.sin(np.pi * np.minimum(np.arange(n_steps) * time_step / pull_duration, 1))**2

    @jax.jit
    def step_objective(trial: Antivector, prediction: Antivector, strength: float) -> tuple[Scalar, Vector]:
        """Incremental elastic, external, and inertial energy and control-space forces at `trial`."""
        energy, force = core.potential(trial, sampling, gradients, sample_volumes,
                                       lame_lambda, lame_mu, clamp, load * strength)
        # Inertia resists departing from the free-flight prediction.
        departure = trial - prediction                                               # Antivector-[n_controls]
        inertial_force = mass(departure) / time_step**2                              # Vector+[n_controls]
        return energy + (inertial_force & departure).sites.sum() / 2, force + inertial_force

    def trajectory() -> Iterator[Antivector]:
        """Yields the surface vertex positions (`[V] Antivector-`) every `n_steps_per_frame` steps."""
        context = JaxContext(ga)
        state = mv.vector([[0, 0, 0]] * n_controls).dual().field()                   # Antivector-[n_controls]
        velocity = state                                                             # Antivector-[n_controls]
        yield vertices + surface_sampling(state)
        for step_index, strength in enumerate(pulse_strengths, 1):
            prediction = (state + velocity * time_step).materialize(context)         # Antivector-[n_controls]

            def objective(trial: Antivector) -> tuple[Scalar, Vector]:
                return tuple(value.materialize(mv.context)
                             for value in step_objective(trial.materialize(context), prediction, strength))

            next_state = minimized_state(state, objective, mass / time_step**2,
                                         n_iterations, tolerance)                    # Antivector-[n_controls]
            velocity = (next_state - state) / time_step                              # Antivector-[n_controls]
            state = next_state                                                       # Antivector-[n_controls]
            if step_index % n_steps_per_frame == 0:
                yield vertices + surface_sampling(state)

    return trajectory(), topology


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    """Streams the Stanford bunny ear-pull simulation to `plots/freeform_bunny.gif` and checks recoil and clamping."""
    from examples.util.animation import save_animation
    from examples.mechanics.freeform import render

    states, topology = bunny_motion()
    V, E, F = topology.n_elements
    n_frames = n_steps // n_steps_per_frame + 1
    recorded_states = []

    def streamed_states() -> Iterator[Antivector]:
        for state in states:
            recorded_states.append(state)
            yield state

    save_animation(render.surface_frames(topology, streamed_states()), "freeform_bunny", frame_duration_ms)

    # checks
    # The ear follows the pull and springs back after release, while the base moves less than a tenth as far.
    surfaces = stack(recorded_states)                                                # [n_frames, V] Antivector-
    displacement = surfaces - surfaces[0]                                            # [n_frames, V] Antivector-
    pull_work = pull_force & displacement                                            # [n_frames, V] Scalar+
    distances = displacement.antinorm()                                              # [n_frames, V] Scalar+
    base_mask = (mv.z & surfaces[0]) < 0.05                                          # [V]
    assert (pull_work.max() > 0.3).all()
    assert (pull_work[-1].max() < 0.05 * pull_work.max()).all()
    assert (distances[:, base_mask].max() < 0.1 * distances.max()).all()


if __name__ == "__main__":
    main()
