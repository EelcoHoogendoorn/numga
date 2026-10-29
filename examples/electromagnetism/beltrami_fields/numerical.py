"""The Beltrami eigensolve removes zero-energy directions before diagonalizing helicity."""

import numpy as np
from scipy.linalg import eigh

from examples.electromagnetism.beltrami_fields.core import Energy, Scalar, as_scalar


def coexact_helicity_eigh(stiffness: Energy, helicity: Energy) -> tuple[Scalar, Scalar, int]:
    """The inverse Beltrami eigenvalues `mu = 1 / lambda`, the energy-orthonormal coexact eigenmodes, and the gradient nullity."""
    n_dofs = stiffness.shape[0]
    s_mat = np.zeros((n_dofs, n_dofs))
    s_mat[stiffness.rows, stiffness.columns] = stiffness.cells.kernel[:, 0, 0]
    h_mat = np.zeros((n_dofs, n_dofs))
    h_mat[helicity.rows, helicity.columns] = helicity.cells.kernel[:, 0, 0]

    evals_s, evecs_s = eigh(s_mat)
    scale = float(evals_s.max())
    coexact = evals_s > 1e-10 * scale
    n_null = int(np.sum(~coexact))
    basis = evecs_s[:, coexact] / np.sqrt(evals_s[coexact])[None, :]
    h_reduced = basis.T @ h_mat @ basis
    mu, vectors = eigh(h_reduced)
    chiral = np.abs(mu) > 1e-8 * float(np.abs(mu).max())
    free_modes = as_scalar((basis @ vectors[:, chiral]).T).field()           # [modes] Scalar+[free]
    return as_scalar(mu[chiral]), free_modes, n_null
