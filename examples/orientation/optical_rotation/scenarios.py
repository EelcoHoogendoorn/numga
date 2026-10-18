"""Polarized light out and back through quartz and through a Faraday rotator, and the mirrored setups."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.orientation.optical_rotation import core
from examples.orientation.optical_rotation.core import Mirror, Polarization, mv

LENGTH = 2.0                                          # mm through the medium
SAMPLES = 9                                           # polarization samples along each leg
ROTARY_POWER = np.deg2rad(22.5)                       # rad per mm; quartz turns 21.7 degrees per mm
VERDET = np.deg2rad(22.5)                             # rad per mm per tesla, illustrative
FIELD = 1.0                                           # tesla, along the path


def isolator() -> tuple[Polarization, np.ndarray]:
    """Light polarized along x travels along z through each medium, reflects at the far end and comes
    back: `[media, setups, legs, depths]` polarizations, the setups the original and its mirror image.

    Both media turn the polarization by 45 degrees on the way out. On the way back the quartz turns it
    back, so the light leaves as it entered; the field turns it on, so the light leaves crossed with how
    it entered, and the polarizer it entered through stops it: an optical isolator.
    """
    direction = mv.z
    start = mv.x
    # Measured against the lab's handedness, the basis volume x ^ y ^ z.
    rotary_power = mv(core.RotaryPower, [ROTARY_POWER])                           # [] RotaryPower
    field = mv.xy * FIELD                                                         # [] Field, along z
    mirror: Mirror = mv.x                                                         # the plane x = 0, holding the path
    depths = np.linspace(0, LENGTH, SAMPLES)

    original = core.through_media(direction, rotary_power, field, VERDET, start, depths)
    # The mirror image of the setup: the medium, the field and the incoming light, all reflected.
    mirrored = core.through_media(direction, mirror >> rotary_power, mirror >> field, VERDET, mirror >> start, depths)
    polarizations = stack([original, mirrored], axis=1)                          # [media, setups, legs, depths] Polarization

    # --- checks
    # The quartz returns the light as it entered; the field returns it crossed.
    active, faraday = original[:, 1, -1]
    assert (active - start).norm().to_array() < 1e-12
    assert (faraday | start).norm().to_array() < 1e-12
    # The mirrored setup is the mirror image of the original, at every depth on both legs.
    assert (mirrored - (mirror >> original)).norm().to_array().max() < 1e-12
    return polarizations, depths


if __name__ == "__main__":
    from examples.animation import save_figure
    from examples.orientation.optical_rotation import render

    save_figure(render.draw_isolator(*isolator()), "optical_rotation")
