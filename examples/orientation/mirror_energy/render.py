"""GPU rendering draws a spherical mass distribution and its mirror image on translucent unit spheres with great
circles, closed orbital paths, connecting rods, and tangent velocity arrows."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import concatenate, stack

from examples.orientation.mirror_energy.core import Point
from numga.topology import Topology
from numga.render import Scene, simplices, solids
from examples.util.panels import image_figure

mass_colours = np.array([to_rgb(colour) for colour in ("#df7246", "#238c9b", "#7662a1")])
sphere_colour, circle_colour, shape_colour = (np.array(to_rgb(colour)) for colour in ("#e6ebf0", "#aab4bf", "#75808a"))
# Viewing angles in radians (azimuth zero looks along +y within the mirror plane x = 0 so reflection flips left-right),
# panel margin, per-case pixel size, opacities, stroke/rod/ball sizes in pixels, and velocity arrow scale.
elevation, azimuth = np.radians([23.0, 0.0])
margin, n_pixels = 0.04, 300
sphere_opacity, path_opacity = 0.12, 0.8
circle_width, path_width, shape_radius, mass_radius = 1.0, 1.67, 0.67, 5.0
arrow_width, chevron_length = 2.34, 9.4
velocity_length = 0.35


# --- plumbing -------------------------------------------------------------------------
def scenes(paths: solids.Point) -> list[Scene]:
    """Static GPU scenes for each case (`[n_frames, n_cases, n_masses] Point-`) hold the translucent unit sphere, three
    great circles, and closed orbital path tubes."""
    n_frames, n_cases, n_masses = paths.shape
    corners = np.stack(np.meshgrid(*[[-1, 1]] * 3, indexing="ij"), axis=-1).reshape(-1, 3)
    view = solids.view([solids.mv.point(corners)], 0.0, elevation, azimuth, margin, n_pixels)
    sphere = solids.ball(solids.origin, 1.0)[None, None]
    # Coordinate planes cut three great circles on the translucent sphere behind the rods and mass balls.
    circles = solids.Ink(n_masses + 1, solids.half_space(solids.axes), solids.everywhere[None],
                         circle_colour, circle_width, 1.0)
    path_loops = Topology.loops([n_frames] * n_masses)
    path_colours = np.repeat(mass_colours, n_frames, axis=0)                    # [n_masses * n_frames, 3]
    return [Scene(view, np.ones(3), [],
                  [(simplices.tubes(concatenate([paths[:, case, mass] for mass in range(n_masses)]), path_loops,
                                    path_colours, path_width, view, gloss=solids.gloss), path_opacity)],
                  inks=[circles], translucent=[(sphere, sphere_colour, sphere_opacity)])
            for case in range(n_cases)]


def case_image(scene: Scene, places: solids.Point, velocities: solids.Point, masses: np.ndarray) -> np.ndarray:
    """Renders one case at one instant (`[n_masses] Point-`), drawing the mass balls, connecting rods, and tangent
    velocity arrows inside `scene`."""
    view = scene.view
    n_masses = len(places)
    # Connecting rods between consecutive masses make the reflected handedness visible throughout the turn.
    next_places = places[(np.arange(n_masses) + 1) % n_masses]                  # [n_masses] Point-
    shape_rods = solids.rods(places, next_places, shape_radius * view.pixel)
    # Each mass is drawn as a ball whose projected area scales with its mass.
    balls = [solids.ball(places[mass:mass + 1], mass_radius * np.sqrt(masses[mass]) * view.pixel)
             for mass in range(n_masses)]
    tips = places + velocity_length * velocities                                # [n_masses] Point-
    arrows = simplices.tubes(stack((places, tips), axis=-1).reshape(-1), Topology.chains([2] * n_masses),
                             mass_colours, arrow_width, view, gloss=solids.gloss, chevrons=np.ones(n_masses),
                             chevron=chevron_length)
    parts = [(shape_rods, shape_colour)] + [(ball[:, None], colour) for ball, colour in zip(balls, mass_colours)]
    return scene.draw(parts, [], [], [(arrows, 1.0)])


def frames(positions: Point, velocities: Point, masses: np.ndarray, moments: range) -> Iterator[np.ndarray]:
    """Yields side-by-side RGB frames of the body and its mirror image (`[n_frames, n_cases, n_masses] Point-`) at
    `moments`."""
    # Embed the spherical positions at unit weight and the tangent velocities as ideal directions in PGA3D.
    places = solids.origin + solids.context.include_from_algebra(positions)     # [...] Point-
    reaches = solids.context.include_from_algebra(velocities)                   # [...] Point-
    case_scenes = scenes(places)
    for time in moments:
        yield np.concatenate([case_image(scene, places[time, case], reaches[time, case], masses)
                              for case, scene in enumerate(case_scenes)], axis=1)


def motion_figure(positions: Point, velocities: Point, masses: np.ndarray) -> plt.Figure:
    """Renders the initial still figure of the turning masses and their mirror image."""
    return image_figure(next(frames(positions, velocities, masses, range(1))))


def motion_animation(positions: Point, velocities: Point, masses: np.ndarray) -> Iterator[np.ndarray]:
    """Yields RGB frames across every sampled instant of the spherical motion and its mirror image."""
    return frames(positions, velocities, masses, range(positions.shape[0]))
