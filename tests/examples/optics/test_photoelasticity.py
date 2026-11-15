"""The circular hole satisfies mechanical equilibrium and its optical response obeys the stress-optic law."""

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.optics.photoelasticity import core, render


round_off = 1e-10
basis_cuts = stack([core.mv.x, core.mv.y])
basis_fields = stack([core.mv.yz, core.mv.zx, core.mv.xy, core.mv.xt, core.mv.yt, core.mv.zt])


def uniform_stress(along_x: float, along_y: float, shear: float) -> core.Stress:
    """Symmetric uniform in-plane stress built from normal components along `x` and `y` and `shear`."""
    x_dir, y_dir = core.mv.x.dual(), core.mv.y.dual()
    return (along_x * x_dir * (core.Cut & x_dir)
            + along_y * y_dir * (core.Cut & y_dir)
            + shear * (x_dir * (core.Cut & y_dir) + y_dir * (core.Cut & x_dir)))


def planar_directions(coordinates: list) -> core.Planar:
    """Planar positions (`[...] Planar-`) constructed from `[..., 2]` Cartesian coordinates."""
    values = np.array(coordinates)
    return core.mv.x.dual() * values[..., 0] + core.mv.y.dual() * values[..., 1]


def stress_difference_norm(difference: core.Stress) -> core.Scalar:
    """Real magnitude of `difference` evaluated on `basis_cuts` (`[...] Scalar+`)."""
    tractions = difference[..., None](basis_cuts)
    return tractions.real().antinorm().real() + (-1j * tractions).real().antinorm().real()


def optical_difference_norm(difference: core.OpticalMap) -> core.Scalar:
    """Real magnitude of `difference` across all basis bivector components (`[...] Scalar+`)."""
    entries = difference(basis_fields[None, :]) & basis_fields[:, None].dual()
    return entries.real().abs() + (-1j * entries).real().abs()


def test_free_rim_and_uniaxial_stress_concentration():
    radius = 1.4
    tension = 2.3
    n_samples = 8
    angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    cuts = (core.mv.xy * (-angles / 2)).exp() >> core.mv.x
    normal = cuts.dual()
    remote = uniform_stress(1.7, -0.4, 0.6)
    axial = uniform_stress(tension, 0.0, 0.0)
    tangent_cuts = core.quarter_turn >> basis_cuts

    traction = core.kirsch_stress(radius * normal, radius, remote)(cuts)
    rim = core.kirsch_stress(radius * basis_cuts.dual(), radius, axial)
    hoop = (tangent_cuts & rim(tangent_cuts)).real()
    expected = tension * np.array([-1, 3])

    assert (traction.antinorm().real() < round_off).all()
    assert ((hoop - expected).abs() < round_off).all()


def test_stress_rotates_with_load_and_positions_and_recovers_remote_field():
    radius = 0.8
    far_distance = 1e7 * radius
    rotor = (core.mv.xy * -0.37).exp()
    positions = planar_directions([[1.2, 0.4], [-1.6, 0.7], [0.3, -2.1]])
    remote = uniform_stress(2.1, 0.3, -0.5)
    far_positions = positions / positions.antinorm() * far_distance
    moved_remote = rotor >> remote(rotor << core.Cut)
    cuts = stack([core.mv.x + 0.3 * core.mv.y, core.mv.y - 1.2 * core.mv.x])

    stress = core.kirsch_stress(positions, radius, remote)
    moved_stress = core.kirsch_stress(rotor >> positions, radius, moved_remote)
    rotated_stress = rotor >> stress(rotor << core.Cut)
    far_stress = core.kirsch_stress(far_positions, radius, remote)
    form = core.Cut & stress(core.Cut)

    assert (stress_difference_norm(moved_stress - rotated_stress) < round_off).all()
    assert ((form(cuts[0], cuts[1]) - form(cuts[1], cuts[0])).real().abs() < round_off).all()
    assert (stress_difference_norm(far_stress - remote) < round_off).all()


def test_mean_stress_delays_every_direction_alike():
    delay_per_stress = 83 * np.pi + 0.41
    mean = 3.7
    base = uniform_stress(2.4, -0.8, 0.5)
    isotropic = uniform_stress(mean, mean, 0.0)
    plates = core.retarder(stack([base, base + isotropic, isotropic]), delay_per_stress)
    delay = np.exp(1j * delay_per_stress * mean)
    x_dir, y_dir = core.mv.x.dual(), core.mv.y.dual()
    unstressed = core.polarizer(x_dir, x_dir) + core.polarizer(y_dir, y_dir)

    assert (optical_difference_norm(plates[1] - delay * plates[0]) < round_off).all()
    assert (optical_difference_norm(plates[2] - delay * unstressed) < round_off).all()


def test_linear_and_circular_dark_fields_match_the_retardance_law():
    n_orientations = 5
    angles = np.linspace(0, np.pi, n_orientations, endpoint=False)
    retardances = np.array([0.0, 0.01, 0.83, np.pi, 2 * np.pi, 7.1, 80 * np.pi + 0.4])
    rotors = (core.mv.xy * (-angles[:, None] / 2)).exp()
    base = uniform_stress(1.0, 0.0, 0.0)
    stress = (rotors >> base(rotors << core.Cut)) * retardances[None, :]

    plates = core.retarder(stress, 1.0)
    linear_power = core.linear_polariscope(plates, core.mv.x.dual()).real()
    circular_power = core.circular_polariscope(plates, core.mv.x.dual()).real()
    phase_intensity = np.sin(retardances / 2)**2
    expected_linear = np.sin(2 * angles[:, None])**2 * phase_intensity
    expected_circular = np.broadcast_to(phase_intensity, (n_orientations, len(retardances)))

    assert ((linear_power - expected_linear).abs() < round_off).all()
    assert ((circular_power - expected_circular).abs() < round_off).all()


def test_loading_cycle_returns_to_dark_and_matches_the_loaded_plate():
    radius = 0.8
    delay_per_stress = 13.7
    loads = np.array([0.0, 0.5, 1.0, 0.5, 0.0])
    positions = planar_directions([[[1.2, 0.4], [-1.6, 0.7], [0.3, -2.1]]])
    remote = uniform_stress(2.1, 0.3, -0.5)

    stress = core.kirsch_stress(positions, radius, remote)
    frames = core.circular_polariscope(core.retarder(stress * loads[:, None, None], delay_per_stress),
                                       core.mv.x.dual()).real()
    scaled_stress = core.kirsch_stress(positions[None], radius, remote * loads[:, None, None])
    expected = core.circular_polariscope(core.retarder(scaled_stress, delay_per_stress), core.mv.x.dual()).real()

    assert ((frames - expected).abs() < round_off).all()
    assert ((frames - frames[::-1]).abs() < round_off).all()
    assert (frames[0].abs() < round_off).all()


def test_render_stress_and_polariscopes():
    centres = np.array([-2.0, -0.7, 0.7, 2.0])
    positions = core.mv.x.dual() * centres[None, :] + core.mv.y.dual() * centres[:, None]
    remote = uniform_stress(10.0, 0.0, 0.0)
    stress = core.kirsch_stress(positions, 1.0, remote)
    principal, _ = core.principal_stresses(stress)
    difference = principal[..., 1] - principal[..., 0]
    plate = core.retarder(stress, 0.1)
    circular = core.circular_polariscope(plate, core.mv.x.dual())

    figure = render.stress_figure(positions, difference, 1.0, 10.0)
    figure.canvas.draw()
    plt.close(figure)

    figure = render.polariscope_figure(positions, stack([circular, circular]), 1.0)
    figure.canvas.draw()
    plt.close(figure)

    frame = next(render.polariscope_animation(positions, [circular], 1.0))
    assert frame.shape == (4, 4, 3)
