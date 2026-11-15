"""Coverage and spacetime ray-tracing checks for the primary and secondary rainbows."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.optics.rainbow import core, render, scenarios


def test_internal_bounces_reflect_across_surface_and_keep_chord_angles() -> None:
    """Turning an internal wavevector by `bounce_rotor` matches specular reflection across the rim
    normal, and every chord meets the drop surface at equal incidence angles at both ends."""
    offsets = scenarios.up * np.linspace(0.1, 0.95, 7)                        # [7] Direction-
    vertex = core.entry_point(scenarios.forward, offsets)                     # [7] Direction-
    inward_face = (-vertex).dual_inverse()                                    # [7] Plane+
    inside_wave = core.refracted_wavevector(core.sunlight_wave, inward_face, np.array(1.333))   # [7] Wavevector+
    turn = core.bounce_rotor(vertex, inside_wave)                             # [7] Rotor+
    next_vertex = turn >> vertex                                              # [7] Direction-
    next_face = next_vertex.dual_inverse()                                    # [7] Plane+
    residual = (turn >> inside_wave) - (next_face >> inside_wave)             # [7] Wavevector+
    assert residual.cast(core.Plane).norm().max() < 1e-12
    assert (residual & core.time).abs().max() < 1e-12
    assert ((inside_wave | inward_face) - (inside_wave | next_face)).abs().max() < 1e-12


def test_sky_figure_and_one_sweep_frame_render() -> None:
    """Both `sky_figure` and one streamed frame of `sweep_animation` render without error."""
    offsets = scenarios.up * np.array([0.3, 0.86])                            # [2] Direction-
    sight_cosines, flux = core.sky_bows(offsets, scenarios.indices, scenarios.n_reflections)   # [2, 31, 2] Scalar+, Scalar+
    sky = render.sky_figure(sight_cosines, flux, scenarios.wavelengths, scenarios.sun_elevation, scenarios.sun_width)
    plt.close(sky)
    bundle = core.ray_bundle(offsets, scenarios.drawn_index, scenarios.n_reflections)
    states = core.swept_ray_states(offsets[:1], scenarios.drawn_index, scenarios.n_reflections)
    frame = next(render.sweep_animation(bundle, offsets, sight_cosines, scenarios.wavelengths, states))
    assert frame.ndim == 3 and frame.shape[-1] == 3
