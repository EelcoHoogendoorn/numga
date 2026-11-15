"""Coverage test for induced surface charge relaxation and rendering."""
import matplotlib.pyplot as plt
from numga import stack, where
from examples.electromagnetism.induced_charge import core, render, scenarios


def test_charge_relaxation():
    topology, vertices, fractions, memberships, boundary = scenarios.conductor_scene(9, 2)
    metal = fractions.sum(axis=0)                                   # Scalar+[V]
    states = list(core.relaxation_states(
        topology, vertices, fractions, boundary, scenarios.permittivity,
        scenarios.conductivities, scenarios.body_charges, scenarios.frame_step, 1))
    frames = list(render.relaxation_animation(topology, vertices, metal, states[:1]))
    assert len(frames) == 1
    assert frames[0].shape == (render.n_pixels, render.n_pixels, 3)

    charges = stack([state[1] for state in states])                 # [2] Scalar+[V]
    net_charge = (charges[:, None] * memberships).sites.sum()       # [2, 2] Scalar+
    negative_charge = (where(charges < 0, -charges, 0)[:, None] * memberships).sites.sum()  # [2, 2] Scalar+
    for figure in (
        render.setup_figure(topology, vertices, metal),
        render.balance_figure(net_charge, negative_charge, scenarios.frame_step),
        render.corner_density_figure(vertices, states[-1][2], scenarios.rod_centre,
                                     scenarios.rod_turn, scenarios.rod_lengths, scenarios.interface_width),
    ):
        plt.close(figure)

    assert ((charges[-1].sites.sum() - scenarios.body_charges.sum()).abs() < 1e-8).all()
    assert ((net_charge[0] - scenarios.body_charges).abs() < 1e-8).all()
