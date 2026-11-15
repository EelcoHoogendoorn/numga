"""Angular emission rates of beta electrons from polarized cobalt-60 separate an experiment from its mirror image.

Three-dimensional geometric algebra models emission directions as intrinsic antivectors and nuclear spin as an intrinsic circulating patch dual to the polarizing coil plane. Pairing the nuclear spin with an emission direction, `spin & direction`, produces an intrinsic scalar, so multiplying by an intrinsic weak-decay asymmetry yields an ordinary scalar emission rate. Reflecting the apparatus across a mirror plane reflects the coil's spin without reversing the weak interaction's intrinsic asymmetry constant, so the mirrored apparatus emits opposite to the optical mirror image.
"""

from __future__ import annotations

from numga import NumpyContext, stack
from numga.algebras import VGA3D

ga = VGA3D
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                       # Scalar+
# Emission directions are intrinsic antivectors.
Direction = ga.gatype.antivector()                                                # Antivector-
# Mirror planes are extrinsic vectors.
Plane = ga.gatype.vector()                                                        # Vector+
# Nuclear spin is the intrinsic angular-momentum patch around which the polarizing coil current circulates.
Spin = ga.gatype.antibivector()                                                   # Antibivector-
# Weak-decay parity asymmetry is an intrinsic scalar measured against a volume orientation.
Asymmetry = ga.gatype.antipseudoscalar()                                          # Antipseudoscalar-


# --- math -----------------------------------------------------------------------------
def emission_rates(spin: Spin, directions: Direction, asymmetry: Asymmetry) -> Scalar:
    """Polarized nuclei of `spin` (`[] Spin-`) emit electrons along unit `directions`
    (`[...] Direction-`) with relative intensity `1 + asymmetry * (spin & directions)` (`[...] Scalar+`)."""
    # Pairing two intrinsic elements produces an intrinsic scalar that multiplies `asymmetry` into an ordinary rate.
    return 1 + asymmetry * (spin & directions)                                    # [...] Scalar+


def parity_cases(spin: Spin, directions: Direction, asymmetry: Asymmetry, mirror: Plane) -> tuple[Spin, Scalar]:
    """Comparing the original setup, its optical mirror image across `mirror`, and the physically
    mirrored coil under the unchanged weak interaction yields `[3] Spin-` coil spins and
    `[3, n_directions] Scalar+` angular emission rates."""
    original_rates = emission_rates(spin, directions, asymmetry)                  # [n_directions] Scalar+
    # An upright mirror displays along `directions` what the original apparatus emits along `mirror >> directions`.
    mirror_image_rates = emission_rates(spin, mirror >> directions, asymmetry)    # [n_directions] Scalar+
    # Rebuilding the mirrored coil reflects `spin` while the weak interaction retains `asymmetry`.
    mirrored_apparatus_rates = emission_rates(mirror >> spin, directions, asymmetry)  # [n_directions] Scalar+
    case_spins = stack([spin, mirror >> spin, mirror >> spin])                    # [3] Spin-
    return case_spins, stack([original_rates, mirror_image_rates, mirrored_apparatus_rates])  # [3, n_directions] Scalar+
