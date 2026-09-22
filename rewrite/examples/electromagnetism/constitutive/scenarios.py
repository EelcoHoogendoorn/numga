"""Scenarios and canonical figure generation for electromagnetic constitutive maps.

Executes the mathematics in `core` and feeds the results to `render`.
Can be run directly via:
    PYTHONPATH=src:. python -m examples.electromagnetism.constitutive.scenarios
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from examples import PLOT_DIR
from examples.electromagnetism.constitutive import core, render

# Parameters:
EPS_GLASS = 2.25
MU_GLASS = 1.0
BETA_BOOST = 0.3
AXION_ALPHA = 0.4


def dispersion_scenario(
    speeds: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, list[float]]]:
    """Execute 1D dispersion scan along z across all media."""
    if speeds is None:
        speeds = np.linspace(0.05, 1.5, 3001)

    glass = core.isotropic_medium(EPS_GLASS, MU_GLASS)
    axion = core.axion_medium(glass, AXION_ALPHA)
    crystal = core.crystal_medium(eps_x=2.25, eps_y=1.5, eps_z=1.5, mu=1.0)
    ferrite = core.ferrite_medium(eps=2.25, mu_inv_x=1.0, mu_inv_y=0.5, mu_inv_z=1.0)
    moving_down = core.boosted_medium(glass, beta=BETA_BOOST, direction=core.z)
    moving_up = core.boosted_medium(glass, beta=BETA_BOOST, direction=-core.z)

    media_dict = {
        "glass at rest": (glass, core.z),
        "glass with axion term": (axion, core.z),
        "crystal along z": (crystal, core.z),
        "ferrite along z": (ferrite, core.z),
        "glass moving with wave": (moving_down, core.z),
        "glass moving against wave": (moving_up, core.z),
    }

    curves = {}
    for name, (medium, k_dir) in media_dict.items():
        svals = core.solve_dispersion_scan(speeds, medium, direction=k_dir)
        curves[name] = svals

    refractive_index = np.sqrt(EPS_GLASS * MU_GLASS)
    with_flow = (1.0 / refractive_index + BETA_BOOST) / (1.0 + BETA_BOOST / refractive_index)
    against_flow = (1.0 / refractive_index - BETA_BOOST) / (1.0 - BETA_BOOST / refractive_index)

    expected = {
        "glass at rest": [1.0 / refractive_index],
        "glass with axion term": [1.0 / refractive_index],
        "crystal along z": [1.0 / np.sqrt(2.25), 1.0 / np.sqrt(1.5)],
        "ferrite along z": [1.0 / np.sqrt(2.25 * 2.0), 1.0 / np.sqrt(2.25)],
        "glass moving with wave": [with_flow],
        "glass moving against wave": [against_flow],
    }

    return speeds, curves, expected


def polarization_scenario() -> list[tuple[str, float, Any, str]]:
    """Extract physical polarization states for the two birefringent modes in the crystal."""
    crystal = core.crystal_medium(eps_x=2.25, eps_y=1.5, eps_z=1.5, mu=1.0)
    speeds = np.linspace(0.05, 1.5, 3001)
    svals = core.solve_dispersion_scan(speeds, crystal, direction=core.z)
    mins = core.minimum_speeds(speeds, svals, threshold=2e-3)

    modes = []
    colors = ["crimson", "dodgerblue"]
    labels = ["Slow Wave (v_x)", "Fast Wave (v_y)"]

    for i, (v, label, color) in enumerate(zip(mins, labels, colors)):
        k = core.mv.scalar([v]) * core.t + core.z
        pol = core.polarization_eigenmodes(k, crystal)
        modes.append((label, float(v), pol, color))

    return modes


def fresnel_surface_scenario(
    n_angles: int = 72,
) -> tuple[np.ndarray, dict[str, list[list[float]]]]:
    """Compute 2D polar Fresnel wave surfaces v(theta) in the xz-plane."""
    angles = np.linspace(0, 2 * np.pi, n_angles)
    speeds = np.linspace(0.4, 1.0, 601)

    glass = core.isotropic_medium(EPS_GLASS, MU_GLASS)
    crystal = core.crystal_medium(eps_x=2.25, eps_y=1.5, eps_z=1.5, mu=1.0)
    moving = core.boosted_medium(glass, beta=0.35, direction=core.z)

    surfaces = {}
    for name, med in [("glass (rest)", glass), ("crystal", crystal), ("moving glass (beta=0.35)", moving)]:
        res = core.fresnel_wave_surface_1d(med, angles, speeds, threshold=4e-3)
        surfaces[name] = [r[1] for r in res]

    return angles, surfaces


def fresnel_drag_scenario(
    n_betas: int = 15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute upstream and downstream phase speeds across medium velocity beta."""
    betas = np.linspace(-0.6, 0.6, n_betas)
    speeds = np.linspace(0.05, 1.2, 1151)
    v_down, v_up = core.fresnel_drag_velocities(EPS_GLASS, MU_GLASS, betas, speeds)
    return betas, v_down, v_up


def dispersion_scenario_plot(plot_path: Path | str | None = None) -> plt.Figure:
    """Render standalone 1D dispersion resonance notch spectrum."""
    speeds, curves, expected = dispersion_scenario()
    return render.draw_dispersion_figure(speeds, curves, expected, plot_path=plot_path)


def polarization_scenario_plot(plot_path: Path | str | None = None) -> plt.Figure:
    """Render standalone 2D transverse polarization eigenmode quivers."""
    modes = polarization_scenario()
    return render.draw_polarizations_figure(modes, plot_path=plot_path)


def fresnel_surface_scenario_plot(
    plot_path: Path | str | None = None,
    n_angles: int = 72,
) -> plt.Figure:
    """Render standalone polar 2D Fresnel wave normal surfaces."""
    angles, surfaces = fresnel_surface_scenario(n_angles=n_angles)
    return render.draw_fresnel_surface_figure(angles, surfaces, plot_path=plot_path)


def fresnel_drag_scenario_plot(
    plot_path: Path | str | None = None,
    n_betas: int = 15,
) -> plt.Figure:
    """Render standalone relativistic Fresnel drag curve vs Einstein addition."""
    betas, v_down, v_up = fresnel_drag_scenario(n_betas=n_betas)
    return render.draw_fresnel_drag_figure(
        betas, v_down, v_up, eps=EPS_GLASS, mu=MU_GLASS, plot_path=plot_path
    )


def wave_comparison_scenario(plot_path: Path | str | None = None) -> plt.Figure:
    """Render 3D propagating EB vector fields comparing isotropic vs birefringent media."""
    v_slow = 1.0 / np.sqrt(EPS_GLASS)
    v_fast = 1.0 / np.sqrt(1.5)
    glass_modes = [(v_slow, core.x), (v_slow, core.y)]
    crystal_modes = [(v_slow, core.x), (v_fast, core.y)]
    return render.draw_wave_comparison_figure(
        glass_modes, crystal_modes, plot_path=plot_path
    )


def canonical_scenario(
    out_dir: Path | str | None = None,
    plot_path: Path | str | None = None,
) -> dict[str, plt.Figure]:
    """Run all simulations and save each static plot split into its own image file."""
    if out_dir is None:
        out_dir = Path(plot_path).parent if plot_path else PLOT_DIR
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    figs = {
        "birefringence_3d": wave_comparison_scenario(out_dir / "constitutive_birefringence_3d.png"),
        "dispersion": dispersion_scenario_plot(out_dir / "constitutive_dispersion.png"),
        "polarizations": polarization_scenario_plot(out_dir / "constitutive_polarizations.png"),
        "fresnel_surface": fresnel_surface_scenario_plot(out_dir / "constitutive_fresnel_surface.png"),
        "fresnel_drag": fresnel_drag_scenario_plot(out_dir / "constitutive_fresnel_drag.png"),
    }

    if plot_path:
        plot_path = Path(plot_path)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        # Ensure plot_path exists for unit tests expecting a specific file:
        figs["birefringence_3d"].savefig(str(plot_path), bbox_inches="tight")

    return figs


def main(plot_path: Path | str | None = None) -> plt.Figure:
    """Main CLI entry point: saves all static plots split into their own image files."""
    if plot_path is None:
        plot_path = PLOT_DIR / "constitutive.png"
    figs = canonical_scenario(plot_path=plot_path)
    return figs["birefringence_3d"]


if __name__ == "__main__":
    out_file = PLOT_DIR / "constitutive.png"
    main(out_file)
