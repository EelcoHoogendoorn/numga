"""Equilibrium surface, constant-volume section, and contact horizontal lift of an ideal gas before and after the
Legendre transformation to the Helmholtz free energy."""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.thermodynamics.contact_geometry import core

# Dimensionless heat capacity at fixed volume, in units of the gas constant times the number of moles.
heat_capacity = 1.5


# --- math -----------------------------------------------------------------------------
def surfaces(entropies: np.ndarray, volumes: np.ndarray) -> tuple[core.Ambient, core.Ambient]:
    """Equilibrium surface sampled over a grid of entropies and volumes, paired with its Helmholtz Legendre image."""
    states = core.mv.s.dual() * entropies[None, :] + core.mv.v.dual() * volumes[:, None]   # [..., ...] State-
    points = core.equilibrium(states, heat_capacity)                                       # [..., ...] Ambient-
    return points, core.helmholtz(points)


def section(entropies: np.ndarray, volume: float) -> tuple[core.Ambient, core.Ambient, core.Ambient, core.Ambient]:
    """Equilibrium curve at fixed volume, its entropy tangents, and their images under the Helmholtz Legendre map."""
    states = core.mv.s.dual() * entropies + core.mv.v.dual() * volume         # [...] State-
    points = core.equilibrium(states, heat_capacity)                          # [...] Ambient-
    tangents = core.entropy_tangent(states, heat_capacity)                    # [...] Ambient-
    pushforwards = core.helmholtz_pushforward(points)                         # [...] Ambient- <- Ambient-
    return points, tangents, core.helmholtz(points), pushforwards(tangents)


def tangent_stages(
        points: core.Ambient, tangents: core.Ambient,
        transformed_points: core.Ambient, transformed_tangents: core.Ambient,
) -> Iterator[tuple[core.Ambient, core.Ambient, core.Ambient, core.Ambient]]:
    """Yields each point of tangency and its tangent vector along the section before and after the Legendre map."""
    for index in range(points.shape[0]):
        yield points[index], tangents[index], transformed_points[index], transformed_tangents[index]


def horizontal_lift() -> tuple[core.Ambient, core.Ambient]:
    """Lifts a closed rectangle in entropy and temperature horizontally along the contact hyperplanes, accumulating an
    energy gap equal to the enclosed area."""
    n_samples = 17
    entropy_limits = np.array([-0.6, 0.6])
    cold, hot = 0.5, 1.5
    initial_energy = 0.1
    parameters = np.linspace(0, 1, n_samples)
    entropy_span = entropy_limits[1] - entropy_limits[0]
    mv = core.mv
    slice_origin = mv.v.dual() + mv.p.dual()                                  # [] Ambient-
    corners = slice_origin + stack((
        mv.s.dual() * entropy_limits[0] + mv.t.dual() * cold + mv.u.dual() * initial_energy,
        mv.s.dual() * entropy_limits[0] + mv.t.dual() * hot + mv.u.dual() * initial_energy,
        mv.s.dual() * entropy_limits[1] + mv.t.dual() * hot + mv.u.dual() * (initial_energy + hot * entropy_span),
        mv.s.dual() * entropy_limits[1] + mv.t.dual() * cold + mv.u.dual() * (initial_energy + hot * entropy_span),
        mv.s.dual() * entropy_limits[0] + mv.t.dual() * cold + mv.u.dual() * (initial_energy + (hot - cold) * entropy_span),
    ))                                                                        # [5] Ambient-
    path = corners[:-1, None] * (1 - parameters) + corners[1:, None] * parameters   # [4, n_samples] Ambient-
    return path, core.helmholtz(path)


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.thermodynamics.contact_geometry import render

    n_entropies, section_volume, frame_duration_ms = 93, 1.0, 40
    entropies, volumes = np.linspace(-0.8, 1.2, n_entropies), np.array([0.6, 0.8, 1.0, 1.3, 1.7])
    family, transformed_family = surfaces(entropies, volumes)
    points, tangents, transformed_points, transformed_tangents = section(entropies, section_volume)
    save_figure(render.draw(family, transformed_family, points, tangents, transformed_points, transformed_tangents),
                "thermodynamics_legendre")
    save_animation(render.legendre(family, transformed_family, points, tangents, transformed_points,
                                   transformed_tangents,
                                   tangent_stages(points, tangents, transformed_points, transformed_tangents)),
                   "thermodynamics_legendre", frame_duration_ms)

    # checks
    np.testing.assert_allclose((core.contact_element(points) & tangents).kernel, 0, atol=1e-10)
    np.testing.assert_allclose((core.contact_element(transformed_points) & transformed_tangents).kernel, 0, atol=1e-10)


if __name__ == "__main__":
    main()
