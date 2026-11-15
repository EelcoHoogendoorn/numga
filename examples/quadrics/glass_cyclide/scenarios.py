"""An orbited glass Dupin cyclide with a spherical bite stands in front of a red ball on a checkered floor."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.quadrics.glass_cyclide import core
from examples.quadrics.glass_cyclide.core import Point, Quadric, ball, horizon, mv, origin, sphere, torus, translator

# Torus dimensions and off-centre inversion sphere that swells one side of the ring and narrows the other.
torus_radius, torus_tube = 2.5, 0.9
inversion_offset, inversion_radius = 8.0, 6.25
inversion = (translator(mv.x.dual() * inversion_offset) >> sphere(origin, inversion_radius)).normalized()  # [] Sphere+
n_around, n_across = 24, 12
# Bounding centre, bounding radius, tilt, and vertical lift that stand the cyclide above the floor.
cyclide_centre = mv.x.dual() * 2.04                                           # [] Direction-
cyclide_radius = 2.6
cyclide_tilt = np.radians(55.0)
ring_lift = mv.z.dual() * 2.25                                                # [] Direction-
# Spherical bite subtracted from the thick side of the cyclide, opaque ball behind it, and floor radius.
bite_centre = translator(mv.x.dual() * 0.6 + mv.y.dual() * 1.2 + mv.z.dual() * 0.9) >> origin  # [] Point-
bite_radius = 0.9
ball_centre = translator(mv.x.dual() * 0.6 + mv.y.dual() * 3.4 + mv.z.dual() * 1.1) >> origin  # [] Point-
ball_radius = 1.1
floor_radius = 60.0
# Optical material tables ordered as bitten glass cyclide, opaque red ball, and checkered floor.
colours = np.array([[0.9, 0.96, 0.97], [0.85, 0.22, 0.16], [0.88, 0.87, 0.85]])
transparencies = np.array([0.96, 0.0, 0.0])
refractions = np.array([1.5, 1.0, 1.0])
tile_density = 0.8
# Solar lamp direction raised from x toward z by elevation and turned toward y by azimuth.
lamp_elevation = np.radians(49.0)
lamp_azimuth = np.radians(-50.0)
lamp_turn = (mv.xy * (-lamp_azimuth / 2)).exp() * (mv.xz * (-lamp_elevation / 2)).exp()        # [] Rotation+
lamp = lamp_turn >> mv.x.dual()                                                                # [] Direction-
# Orbiting camera parameters and output dimensions.
target = translator(mv.z.dual() * 1.9) >> origin                                               # [] Point-
distance = 11.0
elevation = np.radians(20.0)
half_view = 0.31
image_shape = (450, 720)
animation_shape = (330, 528)
n_frames = 96
frame_duration_ms = 40


# --- math -----------------------------------------------------------------------------
def scene_solids() -> tuple[Quadric, dict]:
    """The scene combines four conformal quadrics (`[4] Sphere+ <- Point-`) into the bitten glass cyclide, the red ball, and the floor."""
    centring = translator(-cyclide_centre)                                                     # [] Motor+
    tilt = (mv.yz * (-cyclide_tilt / 2)).exp()                                                 # [] Rotation+
    ring_motor = translator(ring_lift) * tilt * centring                                       # [] Motor+
    cyclide = inversion >> torus(torus_radius, torus_tube)(inversion << Point)                 # [] Sphere+ <- Point-
    shapes = [
        # Intersecting the placed cyclide with the negated bite ball carves a concave lens out of the thick side.
        [ring_motor >> cyclide(ring_motor << Point), -ball(ring_motor >> sphere(bite_centre, bite_radius))],
        [ball(sphere(ball_centre, ball_radius))],
        [ball(-mv.z)],
    ]
    counts = np.array([len(shape) for shape in shapes])
    ends = np.cumsum(counts)
    quadrics = stack([quadric.cast(Quadric) for shape in shapes for quadric in shape])         # [4] Sphere+ <- Point-
    centres = stack([translator(ring_lift) >> origin, ball_centre, origin])                    # [3] Point-
    tile_x = stack([0 * mv.x, 0 * mv.x, mv.x * tile_density])                                  # [3] OriginPlane+
    tile_y = stack([0 * mv.y, 0 * mv.y, mv.y * tile_density])                                  # [3] OriginPlane+
    return quadrics, dict(
        first=ends - counts,
        end=ends,
        centre=centres,
        radius=np.array([cyclide_radius, ball_radius * 1.01, floor_radius]),
        colour=colours,
        transparency=transparencies,
        refraction=refractions,
        tile_x=tile_x,
        tile_y=tile_y,
    )


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    """The scenario renders the still image and 96-frame orbit animation of the glass cyclide and checks its surface invariants."""
    from examples.util.animation import save_animation, save_figure
    from examples.quadrics.glass_cyclide import render
    from examples.util.panels import image_figure

    tracer = render.shader()
    quadrics, solid_table = scene_solids()
    still_turns, still_eyes = core.orbit_cameras(target, distance, elevation, np.zeros(1))     # [1] Rotation+, Point-
    still_image, = render.camera_frames(
        tracer, quadrics, solid_table, still_turns, still_eyes, lamp, half_view, image_shape,
    )
    save_figure(image_figure(still_image), "glass_cyclide")
    azimuths = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)
    orbit_turns, orbit_eyes = core.orbit_cameras(target, distance, elevation, azimuths)        # [n_frames] Rotation+, Point-
    save_animation(
        render.camera_frames(tracer, quadrics, solid_table, orbit_turns, orbit_eyes, lamp, half_view, animation_shape),
        "glass_cyclide",
        frame_duration_ms,
    )

    # checks
    cyclide = inversion >> torus(torus_radius, torus_tube)(inversion << Point)                 # [] Sphere+ <- Point-
    cyclide_points = inversion >> core.torus_points(torus_radius, torus_tube, n_around, n_across)  # [n_across, n_around] Point-
    surface_residuals = (cyclide(cyclide_points) & cyclide_points) / (horizon & cyclide_points).squared()  # [n_across, n_around] Scalar+
    normals = core.outward_normal(cyclide(cyclide_points), cyclide_points)                     # [n_across, n_around] Direction-
    exterior_points = translator((normals / normals.antinorm()) * 0.01) >> cyclide_points      # [n_across, n_around] Point-
    exterior_forms = (cyclide(exterior_points) & exterior_points) / (horizon & exterior_points).squared()  # [n_across, n_around] Scalar+
    assert (surface_residuals.abs() < 1e-10).all()
    assert (exterior_forms > 0).all()


if __name__ == "__main__":
    main()
