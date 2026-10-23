"""Scenes for gravitational-wave antenna patterns: the two LIGO detectors, at Hanford, Washington, and
Livingston, Louisiana, their responses to waves from every direction on the sky, and Hanford's response
as the wave's polarization turns half around."""

from __future__ import annotations

import numpy as np

from examples.relativity.ligo import core

# The detectors: their names, latitudes and longitudes in degrees, and the azimuths of their two arms,
# in degrees clockwise from north.
NAMES = ["Hanford", "Livingston"]
SITE_LATITUDES = [46.4553, 30.5631]
SITE_LONGITUDES = [-119.4078, -90.7744]
AZIMUTHS = [[324, 234], [252, 162]]
# The sky, by the point on Earth each direction is straight above: latitudes from pole to pole, and twice
# as many longitudes once around.
N_SKY_SAMPLES = 61
# The polarization's half turn in frames, the free masses on the ring, and the strain drawn on the
# ring, far beyond any real one so that it shows.
N_FRAMES = 48
N_MASSES = 24
STRAIN_SHOWN = 0.3
DURATION_MS = 80


# --- math -----------------------------------------------------------------------------
def detectors() -> tuple[core.Rotor, core.Point]:
    """Each detector's site frame `[detectors]` and arms `[detectors, 2]`."""
    sites = core.place(np.radians(SITE_LATITUDES), np.radians(SITE_LONGITUDES))   # [detectors] Rotor
    return sites, core.arms(sites, np.radians(AZIMUTHS))


def sky() -> core.Rotor:
    """The frame of each direction on the sky: its first axis to the direction, the others across."""
    latitudes = np.radians(np.linspace(-90, 90, N_SKY_SAMPLES))
    longitudes = np.radians(np.linspace(-180, 180, 2 * N_SKY_SAMPLES - 1))
    return core.place(latitudes[:, None], longitudes)                        # [latitudes, longitudes] Rotor


def patterns(arms: core.Point, frames: core.Rotor) -> tuple[core.Scalar, core.Scalar]:
    """Each detector's responses to the plus and cross strains of waves from each frame's direction."""
    plus, cross = core.strains(frames)                                       # [latitudes, longitudes] Point <- Point each
    return core.response(arms[:, None, None], plus), core.response(arms[:, None, None], cross)


def polarization(site: core.Rotor, arms: core.Point, frames: core.Rotor, angles: np.ndarray
                 ) -> tuple[core.Scalar, core.Point]:
    """One detector's response to the plus strain over the sky as the polarization turns by each angle,
    and a ring of free masses across a wave from straight overhead, stretched by that strain."""
    turned, _ = core.strains(core.polarized(frames, angles[:, None, None]))  # [frames, latitudes, longitudes] Point <- Point
    # Straight overhead, the axes across the wave are the site's east and north.
    east, north = site >> core.mv.y, site >> core.mv.z                       # [] Plane each
    turns = np.linspace(0, 2 * np.pi, N_MASSES, endpoint=False)
    ring = ((east ^ north) * (-turns / 2)).exp() >> east.dual()              # [masses] Point
    overhead, _ = core.strains(core.polarized(site, angles))                 # [frames] Point <- Point
    return core.response(arms, turned), ring + STRAIN_SHOWN * overhead[:, None](ring)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.relativity.ligo import render

    sites, arms = detectors()
    frames = sky()
    directions, zeniths = frames >> core.mv.x.dual(), sites >> core.mv.x.dual()   # [latitudes, longitudes], [detectors] Point
    to_plus, to_cross = patterns(arms, frames)                               # [detectors, latitudes, longitudes] Scalar each
    sensitivities = core.sensitivity(to_plus, to_cross)
    network = sensitivities.squared().sum(axis=0).square_root()              # [latitudes, longitudes] Scalar
    # The cosine between the two detectors' pairs of responses: minus one where they read the same wave
    # with opposite signs.
    alike = (to_plus[0] * to_plus[1] + to_cross[0] * to_cross[1]) / (sensitivities[0] * sensitivities[1])
    angles = np.linspace(0, np.pi, N_FRAMES, endpoint=False)
    responses, rings = polarization(sites[0], arms[0], frames, angles)        # [frames, latitudes, longitudes] Scalar, [frames, masses] Point
    save_figure(render.draw_patterns(directions, to_plus, to_cross, sensitivities, zeniths, NAMES), "ligo_patterns")
    save_figure(render.draw_network(directions, network, alike, zeniths, NAMES), "ligo_network")
    save_animation(render.animate_polarization(directions, responses, sites[0], arms[0], rings, NAMES[0]),
                   "ligo_polarization", DURATION_MS)

    # checks
    # Straight overhead each detector the sensitivity is one, the arms lying across the wave, and nowhere
    # is it more.
    overhead = core.sensitivity(*(core.response(arms, strain) for strain in core.strains(sites)))
    assert np.abs(render.scalars(overhead) - 1).max() < 1e-12
    assert render.scalars(sensitivities).max() <= 1 + 1e-12
    # A wave along the line halfway between the arms, in their plane, is not seen at all; any frame with
    # its first axis along that line will do.
    halfway = (arms[:, 0] + arms[:, 1]).normalized()                         # [detectors] Point
    blind = core.sensitivity(*(core.response(arms, strain) for strain in core.strains((1 + halfway * ~core.mv.x.dual()).normalized())))
    assert np.abs(render.scalars(blind)).max() < 1e-12
    # Spin two: turning the axes across the wave by an angle mixes plus and cross by twice that angle.
    mixed = to_plus[0] * np.cos(2 * angles)[:, None, None] + to_cross[0] * np.sin(2 * angles)[:, None, None]
    assert np.abs(render.scalars(responses - mixed)).max() < 1e-12
    # The two detectors are built nearly opposite: over most of the sky they read a wave with opposite
    # signs.
    assert np.mean(render.scalars(alike) < -0.5) > 0.9


if __name__ == "__main__":
    main()
