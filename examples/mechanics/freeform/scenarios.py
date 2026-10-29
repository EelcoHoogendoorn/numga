"""A brief pull on one ear of an elastic Stanford bunny, held at its base, then released."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples import MODEL_DIR
from examples.mechanics.freeform import core, geometry, weights
from examples.mechanics.freeform.core import mv, Scalar, Antivector, Vector
from examples.mechanics.freeform.numerics import minimize_state

# The bunny's height, its volume samples and kernel centres, the kernels' radius, and the weight modes.
MESH_PATH = MODEL_DIR / "bunny.obj"
HEIGHT = 2
N_QUADRATURE = 1600
N_CENTERS = 96
RADIUS = 0.3
N_MODES = 9
# The material: its Lamé parameters and density.
LAME_LAMBDA = 20
LAME_MU = 10
DENSITY = 1
# The compliant support at the base: its depth and stiffness.
CLAMP_DEPTH = 0.12
CLAMP_STIFFNESS = 500
# The pull near one ear's tip: where, how wide, the total force, and how long it lasts.
pull_center = mv.vector([-0.25, 0.34, 1.92]).dual()          # Antivector-
PULL_WIDTH = 0.23
pull_force = (mv.x + 0.7 * mv.y) * 0.6                       # Vector+
PULL_DURATION = 0.4
# The time step, the steps and the steps per frame drawn, and the minimizer's iterations and tolerance per step.
TIME_STEP = 0.02
N_STEPS = 200
N_STEPS_PER_FRAME = 2
N_ITERATIONS = 30
TOLERANCE = 1e-7


# --- math -----------------------------------------------------------------------------
def bunny() -> tuple[Iterator[Antivector], geometry.Mesh]:
    """The surface's vertices over the steps, from rest, and the surface's mesh."""
    mesh, vertices, samples, sample_volumes, centers = geometry.load(
        MESH_PATH, HEIGHT, N_QUADRATURE, N_CENTERS
    )
    evaluation, gradient = weights.kernels(samples, centers, RADIUS)
    _, modes = weights.modes(evaluation, gradient, sample_volumes, LAME_LAMBDA, LAME_MU, N_MODES)
    # Four control points around the bunny, the corners of a tetrahedron.
    corners = mv.vector([[-1, -1, 0], [1, -1, 2], [-1, 1, 2], [1, 1, 0]]).dual()   # [4] Antivector-
    n_controls = len(corners) * N_MODES
    sampling, gradients = core.deformation_basis(samples, centers, RADIUS, modes, corners)
    surface, _ = core.deformation_basis(vertices, centers, RADIUS, modes, corners)
    mass = core.mass(sampling, sample_volumes * DENSITY)          # Vector+[controls] <- Antivector-[controls]

    # A compliant support holds the base; the pull is localized near one ear tip.
    clamp = (-(mv.z & samples).squared() / CLAMP_DEPTH**2).exp() * CLAMP_STIFFNESS   # [samples] Scalar+
    profile = (-(samples - pull_center).scalar_norm_squared() / PULL_WIDTH**2).exp()   # [samples] Scalar+
    # Normalized over the solid, so that the profile distributes the stated total force.
    load = profile * pull_force / (profile * sample_volumes).sum(axis=0)   # [samples] Vector+
    loading = np.sin(np.pi * np.minimum(np.arange(N_STEPS) * TIME_STEP / PULL_DURATION, 1))**2

    def trajectory() -> Iterator[Antivector]:
        state = mv.vector([[0, 0, 0]] * n_controls).dual().field()   # Antivector-[controls]
        velocity = state                                         # Antivector-[controls]
        yield vertices + surface(state)
        for strength in loading:
            predicted = state + velocity * TIME_STEP             # Antivector-[controls]

            def objective(trial: Antivector) -> tuple[Scalar, Vector]:
                energy, force = core.potential(trial, sampling, gradients, sample_volumes,
                                               LAME_LAMBDA, LAME_MU, clamp, load * strength)
                # Inertia resists departing from the free-flight prediction.
                change = trial - predicted                       # Antivector-[controls]
                inertial_force = mass(change) / TIME_STEP**2      # Vector+[controls]
                return energy + (inertial_force & change).sites.sum() / 2, force + inertial_force

            following = minimize_state(state, objective, mass / TIME_STEP**2, N_ITERATIONS, TOLERANCE)   # Antivector-[controls]
            velocity = (following - state) / TIME_STEP            # Antivector-[controls]
            state = following                                    # Antivector-[controls]
            yield vertices + surface(state)

    return trajectory(), mesh


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation
    from examples.mechanics.freeform import render

    duration_ms = round(TIME_STEP * N_STEPS_PER_FRAME * 1000)
    states, mesh = bunny()
    surfaces = stack(list(states))                                # [steps + 1, vertices] Antivector-
    save_animation(render.frames(surfaces[::N_STEPS_PER_FRAME], mesh), "freeform_bunny", duration_ms)

    # checks
    # The ear follows the pull and springs back after release, while the base moves a tenth as far.
    displacement = surfaces - surfaces[0]                         # [steps + 1, vertices] Antivector-
    pulled = (pull_force & displacement).to_array()               # [steps + 1, vertices]
    distances = displacement.scalar_norm_squared().to_array() ** 0.5   # [steps + 1, vertices]
    base = (mv.z & surfaces[0]) < 0.05                 # [vertices]
    assert pulled.max() > 0.3 and pulled[-1].max() < 0.05 * pulled.max()
    assert distances[:, base].max() < 0.1 * distances.max()


if __name__ == "__main__":
    main()
