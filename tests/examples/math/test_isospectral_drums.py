"""Tests verify transplantation invertibility, seam agreement, matched spectra, and drum rendering."""

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.math.isospectral_drums import core, numerics, render, scenarios


def test_transplantation_is_invertible_and_intertwines_every_field() -> None:
    """Transplantation recovers nodal impulses, agrees across tile seams, and commutes with the Dirichlet Laplacian."""
    n_subdivisions = 4
    source_drum = numerics.drum(scenarios.source_indices, scenarios.n_lattice, n_subdivisions)
    target_drum = numerics.drum(scenarios.target_indices, scenarios.n_lattice, n_subdivisions)
    source_neighbors, source_restriction = source_drum[2], source_drum[3]
    target_neighbors, target_restriction, target_assembly = target_drum[2], target_drum[3], target_drum[4]
    n_samples = (n_subdivisions + 1) * (n_subdivisions + 2) // 2
    mixing = core.tile_mixing(scenarios.contributors, scenarios.signs, n_samples)
    transplant = target_assembly(mixing(source_restriction))
    source_laplacian = (4 * core.Scalar - source_neighbors) * n_subdivisions**2
    target_laplacian = (4 * core.Scalar - target_neighbors) * n_subdivisions**2

    n_interior = source_laplacian.shape[0]
    impulses = core.mv.scalar(np.eye(n_interior)[..., None]).field()            # [n_interior] Scalar+[V_source]
    restored = transplant.solve(transplant(impulses))                           # [n_interior] Scalar+[V_source]
    commutator = target_laplacian(transplant) - transplant(source_laplacian)
    seams = target_restriction(transplant) - mixing(source_restriction)
    unit = core.mv.scalar([1])                                                  # [] Scalar+

    # checks
    assert ((restored - impulses).abs() < 1e-9).all()
    assert (commutator.cells(unit).abs() < 1e-9).all()
    assert (seams.cells(unit).abs() < 1e-9).all()


def test_independent_eigensolves_and_renders() -> None:
    """Independent drum eigensolves match in frequency and render cleanly across figures and one vibration frame."""
    n_subdivisions, n_modes = 6, 6
    n_tiles = len(scenarios.source_indices)
    tiles = stack((
        core.lattice_tiles(scenarios.source_indices, scenarios.n_lattice),
        core.lattice_tiles(scenarios.target_indices, scenarios.n_lattice),
    ))                                                                          # [2, n_tiles, 3] Point-
    source_vertices, source_topology, source_neighbors, source_restriction, _, source_extension = numerics.drum(
        scenarios.source_indices, scenarios.n_lattice, n_subdivisions,
    )
    target_vertices, target_topology, target_neighbors, _, target_assembly, target_extension = numerics.drum(
        scenarios.target_indices, scenarios.n_lattice, n_subdivisions,
    )
    source_laplacian = (4 * core.Scalar - source_neighbors) * n_subdivisions**2
    target_laplacian = (4 * core.Scalar - target_neighbors) * n_subdivisions**2
    source_values, source_waves = core.standing_waves(source_laplacian, source_extension, n_subdivisions, n_modes)
    target_values, _ = core.standing_waves(target_laplacian, target_extension, n_subdivisions, n_modes)

    n_samples = source_restriction.shape[0] // n_tiles
    transplant = target_assembly(core.tile_mixing(scenarios.contributors, scenarios.signs, n_samples)(source_restriction))
    target_waves = transplant(source_waves)                                     # [n_modes] Scalar+[V_target]
    residual = target_laplacian(target_waves) - target_waves * source_values    # [n_modes] Scalar+[V_target]

    pieces = core.tile_pieces(source_waves[0], source_restriction, scenarios.contributors, scenarios.signs, 0)  # [4] Scalar+[n_samples]
    local_vertices, local_topology = numerics.triangle(n_subdivisions)
    tiles_fig = render.tiles_figure(tiles)
    pieces_fig = render.pieces_figure(local_vertices, local_topology, pieces, scenarios.exposure)
    plt.close(tiles_fig)
    plt.close(pieces_fig)

    states = core.vibration_states(
        source_waves, source_values, scenarios.mixed_modes, scenarios.mix_amplitudes,
        transplant, source_extension, target_extension, 1,
    )
    frame = next(render.drum_frames((source_vertices, target_vertices), (source_topology, target_topology), tiles, scenarios.exposure, states))

    # checks
    assert ((source_values - target_values).abs() < 1e-8).all()
    assert (residual.abs() < 1e-8).all()
    assert frame.shape == (render.n_pixels, 2 * render.n_pixels, 3)
