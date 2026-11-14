"""Scene parameters configure face-centred cubic slip systems and generate the chart and trajectory animations."""

from __future__ import annotations

from functools import partial

import numpy as np

from numga import stack
from examples.mechanics.slip_systems import core

# Four mutually neighbouring lattice sites of a face-centred cubic crystal, each opposite its slip plane.
atom_positions = [[0, 0, 0], [0, 1, 1], [1, 1, 0], [1, 0, 1]]
n_samples = 1000
# Initial tensile axes spread across the standard triangle between [001], [011], and [111].
start_directions = [
    [0.05, 0.3, 1], [0.15, 0.32, 1], [0.03, 0.5, 1], [0.2, 0.5, 1],
    [0.05, 0.75, 1], [0.3, 0.6, 1], [0.12, 0.62, 1], [0.4, 0.55, 1],
]
n_starts = len(start_directions)
step_slip = 0.01
steepness = 30
n_steps = 500
n_frames = 88
frame_duration_ms = 40
gallery_pixels = 420


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    """Streams the tensile-axis trajectory animation, saves the two-panel stereographic chart, and checks the invariants."""
    from examples.mechanics.slip_systems import render
    from examples.util.animation import save_animation, save_figure

    atoms = core.mv.vector(atom_positions).dual()                             # [4] Direction-
    slips = core.slip_directions(atoms)                                       # [4, 3] Direction-
    planes = core.slip_planes(atoms, slips)                                   # [4] Plane+
    directions = core.disk_directions(n_samples)                              # [n_samples, n_samples] Direction-
    factors = core.schmid_factors(directions, planes, slips)                  # [n_samples, n_samples, 4, 3] Scalar+
    cube_factors = core.schmid_factors(core.mv.z.dual(), planes, slips).abs()   # [4, 3] Scalar+

    start_axes = core.mv.vector(start_directions).normalized().dual()         # [n_starts] Direction-
    schmid = partial(core.schmid_factors, planes=planes, directions=slips)
    states = []
    stream = (states.append(state) or state for state in core.slip_steps(
        start_axes, planes, slips, step_slip, steepness, n_steps,
    ))
    save_animation(
        render.animate_paths(schmid, start_axes, n_steps, n_frames, stream),
        "slip_systems_paths",
        frame_duration_ms,
    )
    stepped_axes, step_amounts = zip(*states)
    axes = stack([start_axes, *stepped_axes])                                 # [n_steps + 1, n_starts] Direction-
    amounts = stack(step_amounts)                                             # [n_steps, n_starts, 4, 3] Scalar+
    save_figure(render.draw_chart(schmid, planes, slips, axes, amounts), "slip_systems")

    # checks
    goal = core.mv.vector([1, 1, 2]).normalized().dual()                      # Direction-
    system_slip = amounts.abs().sum(axis=0).reshape(n_starts, 12)             # [n_starts, 12] Scalar+
    order = system_slip.argsort(axis=-1)                                      # [n_starts, 12]
    two_system_share = (
        system_slip[np.arange(n_starts)[:, None], order[:, -2:]].sum(axis=-1) / system_slip.sum(axis=-1)
    )                                                                         # [n_starts] Scalar+
    assert ((planes[:, None] & slips).abs() < 1e-12).all()
    assert (factors.abs() <= 0.5 + 1e-12).all()
    assert (cube_factors.max() - 1 / np.sqrt(6)).abs() < 1e-12
    assert ((cube_factors - cube_factors.max()).abs() < 1e-12).sum() == 8
    assert ((axes[-1] - goal).norm() < 0.02).all()
    assert (two_system_share > 0.95).all()


def pinned() -> None:
    """Writes the gallery animation of tensile axes rotating through the standard triangle."""
    from examples.mechanics.slip_systems import render
    from examples.util.animation import pin_animation

    atoms = core.mv.vector(atom_positions).dual()                             # [4] Direction-
    slips = core.slip_directions(atoms)                                       # [4, 3] Direction-
    planes = core.slip_planes(atoms, slips)                                   # [4] Plane+
    start_axes = core.mv.vector(start_directions).normalized().dual()         # [n_starts] Direction-
    schmid = partial(core.schmid_factors, planes=planes, directions=slips)
    states = core.slip_steps(start_axes, planes, slips, step_slip, steepness, n_steps)
    pin_animation(
        render.animate_paths_sized(schmid, start_axes, n_steps, n_frames, gallery_pixels, states),
        "slip_systems",
        frame_duration_ms,
    )


if __name__ == "__main__":
    main()
