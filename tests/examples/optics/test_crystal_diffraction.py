"""Solving through the adjugate preserves phases and the finite crystal's scattering pattern."""

import matplotlib.pyplot as plt
import numpy as np

from examples.optics.crystal_diffraction import core, render


def test_deforming_atoms_and_phase_measurements_preserves_scattering():
    n_shells, n_orders, spacing = 1, 1, 1.3
    stretches = np.array([0.8, 1.7, 1.2])
    shears = np.array([-0.3, 0.6, 0.4])
    angles = np.array([0.2, -0.4, 0.7])
    incident = (core.mv.t - core.mv.z) ^ core.mv.x

    atoms, lattice = core.square_lattice(n_shells, n_orders, spacing)
    deformations = core.layer_deformation(stretches, shears, angles)
    positions = deformations[:, None](atoms)
    adjugates = deformations.adjugate()
    reciprocal = adjugates[:, None].solve(lattice)

    transfers = core.transfer_grid(8.0, 9)
    moved_transfers = adjugates[:, None, None].solve(transfers)
    moved_fields = core.scattered_field(positions, moved_transfers, incident)
    reference_field = core.scattered_field(atoms, transfers, incident)
    moved_brightness = core.energy_density(moved_fields, moved_fields)
    reference_brightness = core.energy_density(reference_field, reference_field)

    phases = reciprocal[:, :, None] & positions[:, None, :]
    reference_phases = lattice[:, None] & atoms[None, :]
    peak_fields = core.scattered_field(positions, reciprocal[:, None, :], incident)
    peaks = core.energy_density(peak_fields, peak_fields)

    assert ((moved_brightness - reference_brightness).abs() < 1e-6).all()
    assert ((phases - reference_phases).abs() < 1e-6).all()
    assert ((peaks - 1).abs() < 1e-6).all()

    figure = render.comparison_figure(positions, reciprocal, transfers, moved_brightness)
    figure.canvas.draw()
    plt.close(figure)

    frame_stream = render.diffraction_animation(
        [(positions[0], reciprocal[0], moved_brightness[0])], transfers
    )
    frame = next(frame_stream)
    assert frame.ndim == 3 and frame.shape[2] == 3
    for _ in frame_stream:
        pass
