"""Scenarios for the stressed cube compare the lab, principal, and maximum-shear frames and sweep a
half-turn rotation through the shear plane."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np

from numga import stack
from examples.mechanics.stress import core

mv, Point = core.mv, core.Point
x, y, z = core.axes
# Isotropic Lamé parameters and imposed triaxial stretches plus in-plane xy shear.
lame = 2.0
shear_modulus = 1.0
stretches = np.array([0.05, -0.05, 0.08])
shear_strain = 0.30
# Frame count over half a turn and smooth 25 fps frame duration.
n_frames = 75
frame_duration_ms = 40


# --- math -----------------------------------------------------------------------------
def elastic_state() -> tuple[core.Strain, core.Stress, core.Scalar, core.Point, core.Rotor]:
    """Imposed triaxial stretch and shear define the cube's strain (`[] Plane+ <- Point-`), Cauchy
    stress (`[] Point- <- Plane+`), and ordered principal stresses, directions, and frame rotor."""
    strain = (core.axes * stretches * (core.axes & Point)).sum(axis=0) + shear_strain * (x * (y & Point) + y * (x & Point))   # [] Plane+ <- Point-
    stress = core.cauchy_stress(strain, lame, shear_modulus)                   # [] Point- <- Plane+
    return strain, stress, *core.principal_frame(stress)


def three_frames() -> tuple[tuple[core.Point, core.Point, core.Point, core.Point, core.Point], core.Scalar]:
    """Lab, principal, and eighth-turn maximum-shear rotors (`[3] Rotor+`) place the cube in three
    reference frames alongside the ordered principal stresses."""
    strain, stress, stresses, directions, principal = elastic_state()
    eighth_turn = (mv.xy * (-np.pi / 8)).exp() * principal                     # [] Rotor+
    rotors = stack([mv.rotor(), principal, eighth_turn])                       # [3] Rotor+
    return core.frame_views(strain, stress, directions, rotors), stresses


def turning_stream(n_steps: int) -> Iterator[tuple[core.Point, core.Point, core.Point, core.Point, core.Point]]:
    """Yields `n_steps` cube views turning half a turn in the `xy` shear plane from the principal
    frame."""
    strain, stress, _, directions, principal = elastic_state()
    return core.turning_views(strain, stress, directions, principal, mv.xy, n_steps)


def recorded_states(states: Iterable, out: list) -> Iterator:
    """Yields each item from `states` while appending it to `out` so one streaming pass feeds both
    the saved animation and `# checks`."""
    for state in states:
        out.append(state)
        yield state


def main() -> None:
    import matplotlib.pyplot as plt

    from examples.util.animation import save_animation, save_figure
    from examples.mechanics.stress import render

    views, stresses = three_frames()
    figure = render.frames_figure(views, stresses)
    save_figure(figure, "stress_frames")
    plt.close(figure)
    swept = []
    save_animation(render.turning_frames(stresses, recorded_states(turning_stream(n_frames), swept)),
                   "stress_rotation", frame_duration_ms)

    # checks
    # In the principal frame no face is sheared, the normal tractions match the principal stresses, and stress acts purely along the principal normals.
    _, stress, _, _, principal = elastic_state()
    _, _, seen_normal, seen_shear, _ = views
    assert (seen_shear[1].norm() < 1e-12).all()
    principal_normals = core.axes & seen_normal[1, :3]                         # [3] Scalar+
    assert ((principal_normals[principal_normals.argsort()] - stresses).abs() < 1e-12).all()
    principal_planes = principal >> core.axes                                  # [3] Plane+
    assert ((principal_planes.dual() & stress(principal_planes)).norm() < 1e-12).all()
    # Turning an eighth of a turn from the principal frame reaches the outer Mohr circle's maximum-shear radius on the +x face.
    centre, radius = core.mohr_circle(stresses)
    assert ((seen_shear[2, 0].norm() - radius).abs() < 1e-8).all()
    # Across the streamed rotation, the +x face's normal and shear tractions lie on Mohr's circle.
    swept_normal = stack([state[2][0] for state in swept])                     # [n_frames] Point-
    swept_shear = stack([state[3][0] for state in swept])                      # [n_frames] Point-
    normal_component, shear_component = x & swept_normal, y & swept_shear      # [n_frames] Scalar+, [n_frames] Scalar+
    circle_residual = (normal_component - centre).squared() + shear_component.squared() - radius.squared()   # [n_frames] Scalar+
    assert (circle_residual.abs() < 1e-8).all()


if __name__ == "__main__":
    main()
