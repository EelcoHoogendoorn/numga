"""A planet with a liquid core and a solid shell rings after a blow just below its free surface."""

from __future__ import annotations

from itertools import chain, repeat

import numpy as np

from numga import where
from examples.mechanics.seismic_waves import core

mv = core.mv

core_radius = 0.55
surface_radius = 1.0
radial_boundaries = np.array([core_radius, surface_radius])
n_radial_divisions = 64

# The liquid stores compression energy and has no shear stiffness.
layer_densities = np.array([5.0, 1.0])
layer_lambdas = np.array([5.0, 1.0])
layer_mus = np.array([0.0, 1.0])
# The liquid's shear viscosity; the shell's damping is its viscoelasticity alone.
layer_viscosities = np.array([0.001, 0.0])

# The place of the blow, on the diagonal towards the upper left, where the moon comes in from the frame's corner.
impact_angle = np.radians(45)
impact_normal = (mv.xy * (-impact_angle / 2)).exp() >> mv.y.dual()   # [] Point-
source_depth = 0.05
source_centre = impact_normal * (surface_radius - source_depth)  # [] Point-
source_width = 0.07
source_support = 3.0
source_speed = 0.05
damping_time = 3e-4
# The moon: its radius, its speed, the angle of its approach from the surface's normal, the shares of its normal
# speed it rebounds with and of its tangential speed it keeps, and the frames it takes to come in.
moon_radius = 0.07
moon_speed = 2.0
approach_angle = np.radians(40)
restitution = 0.5
sliding = 0.6
n_approach_frames = 12
dt = 0.00448
n_frames = 100
n_substeps = 8
frame_duration_ms = 40
# The normal modes, of a uniform ball of cork, which does not spread sideways when squeezed, so that its P waves run
# only the square root of two faster than its S waves: its Lamé parameters and density, its rigid motions, the modes
# found, how close two frequencies are when their modes differ by a turn of the ball, the travelling waves such pairs
# make that are shown, the frames of each one's cycle, and the largest displacement they are shown at.
ball_lambda, ball_mu, ball_density = 0.0, 1.0, 1.0
n_rigid_motions = 3
n_modes = 40
pair_tolerance = 1e-3
n_travelling_waves = 12
n_cycle_frames = 24
mode_peak = 0.0006
displacement_scale = 80.0
dilation_scale = 0.1
vorticity_scale = 0.3


def main() -> None:
    from examples.mechanics.seismic_waves import render
    from examples.util.animation import save_animation, save_figure

    mesh, vertices = core.Mesh.concentric_disk(radial_boundaries, n_radial_divisions)
    centres = (mesh.incidence[2, 0].mean(axis=0) * vertices).batch()   # [faces] Point-
    # The liquid core is region zero, the solid shell region one.
    regions = (centres.norm() > core_radius).astype(int)

    density = core.as_scalar(layer_densities[regions]).field()    # Scalar+[faces]
    lame_lambda = core.as_scalar(layer_lambdas[regions])          # [faces] Scalar+
    lame_mu = core.as_scalar(layer_mus[regions])                  # [faces] Scalar+
    material = core.constitutive(lame_lambda, lame_mu)
    viscosity = core.viscous(core.as_scalar(layer_viscosities[regions]))
    operators = core.Operators(mesh, vertices, material, viscosity, density)
    save_figure(render.draw_materials(mesh, vertices, regions), "seismic_waves_materials")

    # The moon comes in at an angle to the normal of the surface where it strikes, and rebounds with part of its normal
    # speed reversed and part of its tangential speed kept; the planet's blow is the momentum the moon gives up.
    incoming = moon_speed * ((mv.xy * (approach_angle / 2)).exp() >> -impact_normal)   # [] Point-
    normal_part = impact_normal * incoming.dot(impact_normal)                    # [] Point-
    outgoing = sliding * (incoming - normal_part) - restitution * normal_part     # [] Point-
    blow = (incoming - outgoing).normalized()                                    # [] Point-
    contact = impact_normal * (surface_radius + moon_radius)                      # [] Point-
    times = n_substeps * dt * np.arange(-n_approach_frames, n_frames + 1)
    moon = contact + where(times < 0, incoming * times, outgoing * times)        # [frames] Point-

    # A short blow just below the free surface.
    offsets = vertices.batch() - source_centre                  # [vertices] Point-
    source_radius_squared = offsets.norm_squared() / source_width**2
    taper = (1 - source_radius_squared / source_support**2).squared()
    gaussian = (-source_radius_squared / 2).exp()
    envelope = where(source_radius_squared < source_support**2, gaussian * taper, 0 * gaussian)
    velocity = (source_speed * envelope * blow).field()         # Point-[vertices]
    displacement = 0 * velocity                                # Point-[vertices]

    # The planet rests while the moon comes in.
    states = chain(repeat((displacement, displacement), n_approach_frames),
                   core.evolve(operators, displacement, velocity, damping_time, dt, n_frames, n_substeps))
    frames = core.reconstruct(operators, states)
    save_animation(render.animate_waves(mesh, vertices, regions, frames, moon, moon_radius, displacement_scale,
                                       dilation_scale, vorticity_scale),
                   "seismic_waves_waves", frame_duration_ms)

    # The modes of a uniform ball of cork, where only their shapes tell them apart.
    ball_regions = np.ones_like(regions)                        # [faces]
    ball = core.Operators(mesh, vertices, core.constitutive(core.as_scalar(ball_lambda), core.as_scalar(ball_mu)),
                          core.viscous(core.as_scalar(0.0)), core.as_scalar(ball_density))
    squared_frequencies, shapes = core.normal_modes(ball, n_rigid_motions + n_modes)
    squared_frequencies, shapes = squared_frequencies[n_rigid_motions:], shapes[n_rigid_motions:]
    # Two modes that differ by a turn of the ball share their frequency; together they make a wave that travels round it.
    frequency_gaps = squared_frequencies[1:] - squared_frequencies[:-1]
    pair_idx = np.flatnonzero(frequency_gaps < pair_tolerance * squared_frequencies[1:])[:n_travelling_waves]
    waves = core.travelling(ball, shapes[pair_idx], shapes[pair_idx + 1], mode_peak, n_cycle_frames)
    save_animation(render.animate_modes(mesh, vertices, ball_regions, waves, displacement_scale),
                   "seismic_waves_modes", frame_duration_ms)


if __name__ == "__main__":
    main()
