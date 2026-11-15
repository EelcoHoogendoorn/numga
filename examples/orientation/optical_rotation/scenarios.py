"""Round-trip polarization setups compare quartz and a Faraday rotator beside their mirror images."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.orientation.optical_rotation import core
from examples.orientation.optical_rotation.core import Field, mv

# Medium length in mm and depth sample count per pass.
path_length = 2.0
n_depths = 9
# Rotary power in rad/mm (22.5 deg/mm gives a 45-degree single-pass turn over 2 mm, close to quartz's 21.7 deg/mm).
quartz_rotary_power = np.deg2rad(22.5)
# Verdet constant in rad/(mm*T) and axial magnetic field strength in tesla, matched for a 45-degree single-pass turn.
verdet_constant = np.deg2rad(22.5)
field_strength = 1.0

# Beam travel along `z`, input polarizer axis along `x`, and mirror plane `x = 0` containing the beam.
propagation_direction = mv.z.dual()                                               # [] Direction-
polarizer_axis = mv.x.dual()                                                      # [] Direction-
mirror = mv.x                                                                     # [] Plane+


# --- math -----------------------------------------------------------------------------
def isolator_comparison() -> tuple[Field, np.ndarray]:
    """Propagating `x`-polarized light out and back along `z` through quartz and a Faraday rotator,
    together with the reflected setups across `x = 0`, yields `[2, 2, 2, n_depths] Field+` electric
    fields and the `[n_depths]` sample depths."""
    incident_field = core.polarization_field(polarizer_axis)                      # [] Field+
    # Transverse beam patch spanned by the two directions across the path.
    cross_section = mv.xy.dual()                                                  # [] CrossSection-
    # Chiral rotary power measured against the basis volume orientation.
    rotary_power = mv(core.RotaryPower, [quartz_rotary_power])                    # [] RotaryPower-
    # Magnetic field in the transverse spatial plane across the path.
    magnetic_field = mv.xy * field_strength                                       # [] Field+
    depths = np.linspace(0, path_length, n_depths)

    original_fields = core.media_round_trips(
        propagation_direction, cross_section, rotary_power, magnetic_field, verdet_constant, incident_field, depths
    )                                                                             # [2, 2, n_depths] Field+
    # Reflecting every geometric and material input across `mirror` flips chiral rotary power while preserving magnetic turn sense.
    mirrored_fields = core.media_round_trips(
        mirror >> propagation_direction, mirror >> cross_section, mirror >> rotary_power,
        mirror >> magnetic_field, verdet_constant, mirror >> incident_field, depths
    )                                                                             # [2, 2, n_depths] Field+
    return stack([original_fields, mirrored_fields], axis=1), depths              # [2, 2, 2, n_depths] Field+


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_figure
    from examples.orientation.optical_rotation import render

    fields, depths = isolator_comparison()
    save_figure(render.isolator_figure(fields, depths), "optical_rotation")

    # checks
    # Quartz unwinds the polarization back to the input state, whereas the Faraday rotator returns it crossed with the polarizer.
    incident_field = core.polarization_field(polarizer_axis)                      # [] Field+
    original_fields, mirrored_fields = fields[:, 0], fields[:, 1]                 # [2, 2, n_depths] Field+, Field+
    chiral_exit, faraday_exit = original_fields[:, 1, -1]                         # [] Field+, Field+
    assert ((chiral_exit - incident_field).scalar_norm_squared().abs() < 1e-12).all()
    assert ((faraday_exit & (polarizer_axis & core.time)).abs() < 1e-12).all()
    # Reflecting all inputs produces the exact reflection of the field at every depth on both legs.
    assert (((mirrored_fields - (mirror >> original_fields)).scalar_norm_squared()).abs() < 1e-12).all()


if __name__ == "__main__":
    main()
