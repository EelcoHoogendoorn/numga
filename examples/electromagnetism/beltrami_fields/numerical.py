"""The Beltrami eigensolve removes zero-energy directions before diagonalizing helicity."""

import numpy as np
from scipy.linalg import eigh

from examples.electromagnetism.beltrami_fields.core import Form, Scalar, mv


def coexact_helicity_eigh(stiffness: Form, helicity: Form) -> tuple[Scalar, Scalar, int]:
    """Diagonalizing `helicity` in the positive-energy subspace of `stiffness` gives inverse eigenvalues, modes, and nullity."""
    n_dofs = stiffness.shape[0]
    stiffness_matrix = np.zeros((n_dofs, n_dofs))
    stiffness_matrix[stiffness.rows, stiffness.columns] = stiffness.cells.kernel[:, 0, 0]
    helicity_matrix = np.zeros((n_dofs, n_dofs))
    helicity_matrix[helicity.rows, helicity.columns] = helicity.cells.kernel[:, 0, 0]

    energy_values, energy_vectors = eigh(stiffness_matrix)
    energy_scale = float(energy_values.max())
    coexact = energy_values > 1e-10 * energy_scale
    n_null = int(np.sum(~coexact))
    orthonormal_basis = energy_vectors[:, coexact] / np.sqrt(energy_values[coexact])[None, :]
    reduced_helicity = np.swapaxes(orthonormal_basis, 0, 1) @ helicity_matrix @ orthonormal_basis
    inverse_values, vectors = eigh(reduced_helicity)
    chiral = np.abs(inverse_values) > 1e-8 * float(np.abs(inverse_values).max())
    mode_table = np.swapaxes(orthonormal_basis @ vectors[:, chiral], 0, 1)
    free_modes = mv.scalar(mode_table[..., None]).field()                       # [...] Scalar+[E_free]
    return mv.scalar(inverse_values[chiral, None]), free_modes, n_null
