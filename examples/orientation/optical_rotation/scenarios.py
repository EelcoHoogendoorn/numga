"""Polarized light out and back through quartz and through a Faraday rotator, and the mirrored setups."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.orientation.optical_rotation import core
from examples.orientation.optical_rotation.core import Field, mv

# The path through the medium, in mm, and the samples of the light along each leg.
length = 2.0
n_samples = 9
# The rotary power in rad per mm; quartz turns 21.7 degrees per mm.
quartz_rotary_power = np.deg2rad(22.5)
# The Verdet constant in rad per mm per tesla, illustrative, and the magnetic field's strength in tesla.
verdet = np.deg2rad(22.5)
field_strength = 1.0

# The light travels along z, through a polarizer along x; the mirror is the plane x = 0, holding the path.
travel = mv.z.dual()                                                              # [] Direction-
polarizer_axis = mv.x.dual()                                                      # [] Direction-
mirror = mv.x                                                                     # [] Plane+


# --- math -----------------------------------------------------------------------------
def isolator() -> tuple[Field, np.ndarray]:
    """Light polarized along x travels along z through each medium, reflects at the far end and comes
    back: `[media, setups, legs, depths]` electric fields, the setups the original and its mirror image.

    Both media turn the light by 45 degrees on the way out. On the way back the quartz turns it back, so the
    light leaves as it entered; the magnetic field turns it on, so the light leaves crossed with how it
    entered, and the polarizer it entered through stops it: an optical isolator.
    """
    entering = core.field_along(polarizer_axis)                                   # [] Field+
    # The beam's cross-section, spanned by the two directions across the path.
    cross_section = mv.xy.dual()                                                  # [] CrossSection-
    # Measured against the lab's handedness, the basis volume x ^ y ^ z.
    rotary_power = mv(core.RotaryPower, [quartz_rotary_power])                    # [] RotaryPower-
    # A plane of space, across the path.
    magnetic_field = mv.xy * field_strength                                       # [] Field+
    depths = np.linspace(0, length, n_samples)

    original = core.through_media(travel, cross_section, rotary_power, magnetic_field, verdet, entering, depths)
    # The mirror image of the setup: the beam, the medium, the magnetic field and the incoming light, all reflected.
    mirrored = core.through_media(mirror >> travel, mirror >> cross_section, mirror >> rotary_power,
                                  mirror >> magnetic_field, verdet, mirror >> entering, depths)
    return stack([original, mirrored], axis=1), depths                           # [media, setups, legs, depths] Field+


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_figure
    from examples.orientation.optical_rotation import render

    fields, depths = isolator()
    save_figure(render.draw_isolator(fields, depths), "optical_rotation")

    # checks
    # The quartz returns the light as it entered; the magnetic field returns it crossed with the polarizer's
    # axis. The mirrored setup is the mirror image of the original, at every depth on both legs.
    entering = core.field_along(polarizer_axis)
    original, mirrored = fields[:, 0], fields[:, 1]
    active, faraday = original[:, 1, -1]
    assert abs((active - entering).scalar_norm_squared().to_array()) < 1e-24
    assert abs((faraday & (polarizer_axis & core.time)).to_array()) < 1e-12
    assert abs((mirrored - (mirror >> original)).scalar_norm_squared().to_array()).max() < 1e-24


if __name__ == "__main__":
    main()
