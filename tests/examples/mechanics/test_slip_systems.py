"""Coverage test steps face-centred cubic slip systems once and renders both the chart and the trajectory animation."""

from functools import partial

import matplotlib.pyplot as plt
from numga import stack

from examples.mechanics.slip_systems import core, render, scenarios


def test_slip_systems_step_and_render():
    atoms = core.mv.vector(scenarios.atom_positions).dual()                   # [4] Direction-
    slips = core.slip_directions(atoms)                                       # [4, 3] Direction-
    planes = core.slip_planes(atoms, slips)                                   # [4] Plane+
    directions = core.disk_directions(2)                                      # [2, 2] Direction-
    factors = core.schmid_factors(directions, planes, slips)                  # [2, 2, 4, 3] Scalar+
    assert (factors.abs() <= 0.5 + 1e-12).all()

    start_axes = core.mv.vector(scenarios.start_directions[:2]).normalized().dual()   # [2] Direction-
    schmid = partial(core.schmid_factors, planes=planes, directions=slips)
    states = []
    stream = (states.append(state) or state for state in core.slip_steps(
        start_axes, planes, slips, scenarios.step_slip, scenarios.steepness, 1,
    ))
    next(render.animate_paths(schmid, start_axes, 1, 1, stream))
    axes = stack([start_axes, states[0][0]])                                  # [2, 2] Direction-
    amounts = stack([states[0][1]])                                           # [1, 2, 4, 3] Scalar+
    figure = render.draw_chart(schmid, planes, slips, axes, amounts)
    plt.close(figure)
