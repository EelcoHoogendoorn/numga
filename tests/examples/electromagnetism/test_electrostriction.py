"""Coverage checks run electrostriction equilibrium, strain readout, and lattice rendering on a one-ring hexagonal patch."""

import matplotlib.pyplot as plt

from examples.electromagnetism.electrostriction import core, render, scenarios


def test_electrostriction_loading_and_rendering() -> None:
    topology, rest_positions = scenarios.reference_lattice(n_rings=1, spacing=scenarios.spacing)
    bond_ends = topology.d[0]

    strengths, load_fields, load_positions, load_dipoles = scenarios.loading_curve(
        rest_positions,
        bond_ends,
        scenarios.field_strength,
        scenarios.polarizability,
        scenarios.stiffness,
        scenarios.tether_stiffness,
        n_load_samples=2,
        n_newton_steps=2,
    )
    parallel_strain = core.axial_strain(load_positions, rest_positions, core.mv.x)
    transverse_strain = core.axial_strain(load_positions, rest_positions, core.mv.y)
    assert ((load_positions[0] - rest_positions).antinorm() < 1e-6).all()
    assert parallel_strain[-1] < 0
    assert transverse_strain[-1] > 0

    plt.close(render.rest_figure(rest_positions, topology))
    plt.close(
        render.lattice_figure(
            rest_positions, topology, load_positions[-1], load_dipoles[-1], load_fields[-1],
            scenarios.dipole_scale, scenarios.field_scale,
        )
    )
    plt.close(render.loading_figure(strengths, parallel_strain, transverse_strain))

    states = core.turning_states(
        rest_positions, bond_ends, scenarios.field_strength,
        scenarios.polarizability, scenarios.stiffness, scenarios.tether_stiffness,
        n_frames=2, n_steps=2,
    )
    frames = list(
        render.lattice_animation(
            rest_positions, topology, scenarios.dipole_scale, scenarios.field_scale, states,
        )
    )
    assert len(frames) == 2
