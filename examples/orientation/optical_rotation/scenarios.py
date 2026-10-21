"""Polarized light out and back through quartz and through a Faraday rotator, and the mirrored setups."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.orientation.optical_rotation import core
from examples.orientation.optical_rotation.core import Mirror, Polarization, mv

LENGTH = 2.0                                          # mm through the medium
N_SAMPLES = 9                                         # polarization samples along each leg
ROTARY_POWER = np.deg2rad(22.5)                       # rad per mm; quartz turns 21.7 degrees per mm
VERDET = np.deg2rad(22.5)                             # rad per mm per tesla, illustrative
FIELD = 1.0                                           # tesla, along the path


# --- math -----------------------------------------------------------------------------
def isolator() -> tuple[Polarization, np.ndarray]:
    """Light polarized along x travels along z through each medium, reflects at the far end and comes
    back: `[media, setups, legs, depths]` polarizations, the setups the original and its mirror image.

    Both media turn the polarization by 45 degrees on the way out. On the way back the quartz turns it
    back, so the light leaves as it entered; the field turns it on, so the light leaves crossed with how
    it entered, and the polarizer it entered through stops it: an optical isolator.
    """
    direction = mv.z.dual()
    start = mv.x.dual()
    # Measured against the lab's handedness, the basis volume x ^ y ^ z.
    rotary_power = mv(core.RotaryPower, [ROTARY_POWER])                           # [] RotaryPower
    field = mv.xy * FIELD                                                         # [] Field, along z
    mirror: Mirror = mv.x                                                         # the plane x = 0, holding the path
    depths = np.linspace(0, LENGTH, N_SAMPLES)

    original = core.through_media(direction, rotary_power, field, VERDET, start, depths)
    # The mirror image of the setup: the medium, the field and the incoming light, all reflected.
    mirrored = core.through_media(direction, mirror >> rotary_power, mirror >> field, VERDET, mirror >> start, depths)
    return stack([original, mirrored], axis=1), depths                           # [media, setups, legs, depths] Polarization


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_figure
    from examples.orientation.optical_rotation import render

    polarizations, depths = isolator()
    save_figure(render.draw_isolator(polarizations, depths), "optical_rotation")

    # checks
    # The quartz returns the light as it entered; the field returns it crossed. The mirrored setup is the
    # mirror image of the original, at every depth on both legs.
    start, mirror = mv.x.dual(), mv.x
    original, mirrored = polarizations[:, 0], polarizations[:, 1]
    active, faraday = original[:, 1, -1]
    assert (active - start).squared().to_array() < 1e-24
    assert abs(faraday.dot(start).to_array()) < 1e-12
    assert (mirrored - (mirror >> original)).squared().to_array().max() < 1e-24


if __name__ == "__main__":
    main()
