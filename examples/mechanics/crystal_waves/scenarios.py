"""Elastic waves in three cubic materials: wave speeds, wave-surface sections, and phonon focusing."""

from __future__ import annotations

import numpy as np

from examples.mechanics.crystal_waves import core

# Cubic elastic constants c11, c12, c44 in GPa and density in g/cm^3, giving wave speeds in km/s.
# Fused silica is isotropic (`c12 == c11 - 2 * c44`), silicon is moderately anisotropic, and beta-brass
# is strongly anisotropic, with Zener ratios `2 * c44 / (c11 - c12)` of 1.0, 1.6 and 8.5.
names = ["fused silica", "silicon", "beta-brass"]
n_materials = len(names)
c11 = np.array([78.5, 165.7, 129.1])
c12 = np.array([16.1, 63.9, 109.7])
c44 = np.array([31.2, 79.6, 82.4])
density = np.array([2.20, 2.33, 7.60])

modes = ("slow shear", "fast shear", "compression")

# Sphere heading count and RNG seed for phonon focusing.
n_headings = 400_000
seed = 0
# In-plane heading count around the z = 0 cube face for wave-surface sections.
n_angles = 4000


def crystals() -> tuple[core.Stiffness, np.ndarray]:
    """Cubic stiffness maps (`[n_materials] Plane+ <- (Plane+, Direction-, Plane+)`) and densities for the three materials."""
    return core.stiffness(c11, c12, c44), density                   # [n_materials] Plane+ <- (Plane+, Direction-, Plane+), [n_materials]


def focusing(n_headings: int, seed: int) -> tuple[core.Direction, core.Plane, core.Scalar]:
    """Group velocities (`[n_materials, n_headings, 3] Direction-`), headings (`[n_headings] Plane+`), and wave moduli (`[n_materials, n_headings, 3] Scalar+`) over the sphere."""
    crystal_stiffnesses, material_densities = crystals()            # [n_materials] Plane+ <- (Plane+, Direction-, Plane+), [n_materials]
    headings = core.mv.vector(np.random.default_rng(seed).normal(size=(n_headings, 3))).normalized()   # [n_headings] Plane+
    wave_moduli, polarizations = core.waves(crystal_stiffnesses[:, None], headings)   # [n_materials, n_headings, 3] Scalar+, [n_materials, n_headings, 3] Direction-
    velocities = core.energy_flow(crystal_stiffnesses[:, None], headings, polarizations, material_densities[:, None])   # [n_materials, n_headings, 3] Direction-
    return velocities, headings, wave_moduli


def wave_fronts(n_angles: int) -> core.Direction:
    """Group velocities (`[n_materials, n_angles, 3] Direction-`) for headings around the z = 0 cube face, ordered by wave speed."""
    crystal_stiffnesses, material_densities = crystals()            # [n_materials] Plane+ <- (Plane+, Direction-, Plane+), [n_materials]
    angles = np.linspace(0.0, 2 * np.pi, n_angles, endpoint=False)
    # Turn the x plane through every angle in the z = 0 face; where shear speeds cross, the rank-ordered modes swap polarizations.
    rotors = (core.mv.xy * (-angles / 2)).exp()                     # [n_angles] Rotor+
    headings = rotors >> core.mv.x                                  # [n_angles] Plane+
    _, polarizations = core.waves(crystal_stiffnesses[:, None], headings)   # [n_materials, n_angles, 3] Scalar+, [n_materials, n_angles, 3] Direction-
    velocities = core.energy_flow(crystal_stiffnesses[:, None], headings, polarizations, material_densities[:, None])   # [n_materials, n_angles, 3] Direction-
    return velocities


def main() -> None:
    from examples.util.animation import save_figure
    from examples.mechanics.crystal_waves import render

    velocities, headings, wave_moduli = focusing(n_headings, seed)
    focusing_fig = render.focusing_figure(velocities, names)
    fronts_fig = render.wave_fronts_figure(wave_fronts(n_angles), names)
    save_figure(focusing_fig, "crystal_waves_focusing")
    save_figure(fronts_fig, "crystal_waves_fronts")
    import matplotlib.pyplot as plt
    plt.close(focusing_fig)
    plt.close(fronts_fig)

    # checks
    crystal_stiffnesses, material_densities = crystals()            # [n_materials] Plane+ <- (Plane+, Direction-, Plane+), [n_materials]
    # Group velocity paired with its heading equals the phase speed.
    phase_speeds = (wave_moduli / material_densities[:, None, None]).square_root()   # [n_materials, n_headings, 3] Scalar+
    assert (((headings[:, None] & velocities) - phase_speeds).abs() < 1e-7 * phase_speeds).all()
    # Closed-form speeds along a cube edge [100] and a face diagonal [110].
    symmetry_headings = core.mv.vector([[1, 0, 0], [1, 1, 0]]).normalized()   # [2] Plane+
    symmetry_moduli, _ = core.waves(crystal_stiffnesses[:, None], symmetry_headings)   # [n_materials, 2, 3] Scalar+, [n_materials, 2, 3] Direction-
    symmetry_speeds = (symmetry_moduli / material_densities[:, None, None]).square_root()   # [n_materials, 2, 3] Scalar+
    edge_speeds = np.sqrt(np.stack([c44, c44, c11], axis=-1) / material_densities[:, None])   # [n_materials, 3]
    diagonal_moduli = np.sort(np.stack([(c11 - c12) / 2, c44, (c11 + c12 + 2 * c44) / 2], axis=-1), axis=-1)   # [n_materials, 3]
    diagonal_speeds = np.sqrt(diagonal_moduli / material_densities[:, None])   # [n_materials, 3]
    assert ((symmetry_speeds[:, 0] - edge_speeds).abs() < 1e-7 * edge_speeds).all()
    assert ((symmetry_speeds[:, 1] - diagonal_speeds).abs() < 1e-7 * diagonal_speeds).all()


if __name__ == "__main__":
    main()
