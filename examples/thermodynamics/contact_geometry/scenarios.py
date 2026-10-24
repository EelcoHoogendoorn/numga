"""The ideal gas's equilibrium surface before and after the Legendre change to the Helmholtz energy."""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.thermodynamics.contact_geometry import core

HEAT_CAPACITY = 1.5


# --- math -----------------------------------------------------------------------------
def surfaces(entropies: np.ndarray, volumes: np.ndarray) -> tuple[core.Ambient, core.Ambient]:
    """The equilibrium surface over entropies and volumes `[volumes, entropies]`, and its Legendre change."""
    states = core.mv.s.dual() * entropies[None, :] + core.mv.v.dual() * volumes[:, None]
    lifted = core.equilibrium(states, HEAT_CAPACITY)
    return lifted, core.helmholtz(lifted)


def section(entropies: np.ndarray, volume: float) -> tuple[core.Ambient, core.Ambient, core.Ambient, core.Ambient]:
    """The equilibrium points at a fixed volume, their changes along the entropy, and both after the Legendre change."""
    states = core.mv.s.dual() * entropies + core.mv.v.dual() * volume
    embedding = lambda state: core.equilibrium(state, HEAT_CAPACITY)
    points = embedding(states)
    tangents = core.derivative(embedding)(states)(core.mv.s.dual())                  # [samples] Ambient
    return points, tangents, core.helmholtz(points), core.derivative(core.helmholtz)(points)(tangents)


def moving(points: core.Ambient, tangents: core.Ambient, changed: core.Ambient,
           changed_tangents: core.Ambient) -> Iterator[tuple[core.Ambient, core.Ambient, core.Ambient, core.Ambient]]:
    """The point of tangency moving along the section, before and after the change."""
    for index in range(points.shape[0]):
        yield points[index], tangents[index], changed[index], changed_tangents[index]


def horizontal_lift() -> tuple[core.Ambient, core.Ambient]:
    """A loop of entropy and temperature at fixed volume and pressure, lifted along the contact planes: every edge
    changes the energy by temperature times the entropy's change, so the lift does not close but gains the area the
    loop encloses."""
    n_samples = 17
    entropy_limits = np.array([-0.6, 0.6])
    cold, hot = 0.5, 1.5
    initial_energy = 0.1
    parameters = np.linspace(0, 1, n_samples)
    entropy_span = entropy_limits[1] - entropy_limits[0]
    mv = core.mv
    slice_origin = mv.v.dual() + mv.p.dual()
    corners = slice_origin + stack((
        mv.s.dual() * entropy_limits[0] + mv.t.dual() * cold + mv.u.dual() * initial_energy,
        mv.s.dual() * entropy_limits[0] + mv.t.dual() * hot + mv.u.dual() * initial_energy,
        mv.s.dual() * entropy_limits[1] + mv.t.dual() * hot + mv.u.dual() * (initial_energy + hot * entropy_span),
        mv.s.dual() * entropy_limits[1] + mv.t.dual() * cold + mv.u.dual() * (initial_energy + hot * entropy_span),
        mv.s.dual() * entropy_limits[0] + mv.t.dual() * cold + mv.u.dual() * (initial_energy + (hot - cold) * entropy_span),
    ))
    path = corners[:-1, None] * (1 - parameters) + corners[1:, None] * parameters
    return path, core.helmholtz(path)


if __name__ == "__main__":
    from examples.util.animation import save_animation, save_figure
    from examples.thermodynamics.contact_geometry import render

    entropies, volumes = np.linspace(-0.8, 1.2, 41), np.array([0.6, 0.8, 1.0, 1.3, 1.7])
    family, changed_family = surfaces(entropies, volumes)
    points, tangents, changed, changed_tangents = section(entropies, 1.0)
    save_figure(render.draw(family, changed_family, points, tangents, changed, changed_tangents),
                "thermodynamics_legendre")
    save_animation(render.legendre(family, changed_family, points, tangents, changed, changed_tangents,
                                   moving(points, tangents, changed, changed_tangents)), "thermodynamics_legendre", 90)
