"""The mathematics of every example imports without the plotting stack."""

import subprocess
import sys

PROBE = """
from numga.algebra import Algebra
from numga.algebras import PGA2D, Spherical3D
from examples import instantiate
import sys
import examples.electromagnetism.constitutive.core
import examples.electromagnetism.fresnel.core
import examples.electromagnetism.fresnel.scenarios
import examples.electromagnetism.moving_charge.core
import examples.electromagnetism.moving_charge.scenarios
import examples.electromagnetism.second_harmonic.core
import examples.electromagnetism.second_harmonic.scenarios
import examples.estimation.odometry.scenarios
import examples.quadrics.cyclides.core
import examples.estimation.epipolar.core
import examples.geometry.fitting.core
import examples.geometry.fitting.scenarios
import examples.estimation.kalman.core
import examples.estimation.kalman.scenarios
import examples.estimation.pose_diffusion.core
import examples.estimation.pose_diffusion.scenarios
import examples.geometry.projection.core
import examples.surfaces.qem.core
import examples.geometry.registration.core
import examples.geometry.registration.scenarios
import examples.geometry.scenegraph.core
import examples.surfaces.surface_curvature.core
import examples.math.extensor_representations.core
import examples.math.extensor_representations.scenarios
import examples.math.hopf.core
import examples.math.hopf.scenarios
import examples.math.klein_quadric.core
import examples.math.klein_quadric.scenarios
import examples.math.pascal.core
import examples.math.pascal.scenarios
import examples.math.poncelet.core
import examples.math.poncelet.scenarios
import examples.math.spin_groups.core
import examples.math.spin_groups.scenarios
import examples.mechanics.crystal_waves.core
import examples.mechanics.area_transport.core
import examples.mechanics.area_transport.scenarios
import examples.mechanics.crystal_waves.scenarios
import examples.mechanics.manipulability.core
import examples.mechanics.modes.core
import examples.mechanics.modal_xpbd.core
import examples.mechanics.modal_xpbd.scenarios
import examples.mechanics.riccati.core
import examples.mechanics.riccati.scenarios
import examples.mechanics.robot_arm.core
import examples.mechanics.spinning_top.core
import examples.mechanics.spinning_top.scenarios
import examples.mechanics.symmetry.core
import examples.mechanics.tides.core
import examples.mechanics.tides.scenarios
import examples.mechanics.vortices.core
import examples.mechanics.vortices.scenarios
import examples.mechanics.wing.core
import examples.mechanics.wing.scenarios
import examples.mechanics.xpbd.core
import examples.optics.lens_camera.core
import examples.optics.crystal_diffraction.core
import examples.optics.crystal_diffraction.scenarios
import examples.optics.thin_lens.core
import examples.quantum.graphene.core
import examples.quantum.graphene.scenarios
import examples.quantum.hubbard_dimer.core
import examples.quantum.hubbard_dimer.scenarios
import examples.quantum.hubbard_ring.core
import examples.quantum.hubbard_ring.scenarios
import examples.quantum.magnetic_resonance.core
import examples.quantum.magnetic_resonance.scenarios
import examples.quantum.process_tomography.core
import examples.quantum.process_tomography.scenarios
import examples.quantum.two_spins.core
import examples.quantum.two_spins.scenarios
import examples.quadrics.cayley_klein.core
import examples.quadrics.cga_spherical_quadrics.core
import examples.quadrics.conformal_elliptical.core
import examples.quadrics.gaussian.core
import examples.quadrics.quadric_collision.core
import examples.quadrics.quadrics.core
import examples.quadrics.s3_raytracer.core
import examples.quadrics.spherical_quadrics.core
import examples.relativity.curvature.core
import examples.relativity.dirac.core
import examples.relativity.dirac.scenarios
import examples.relativity.gravitational_lensing.core
import examples.relativity.gravitational_lensing.scenarios
import examples.relativity.impulse.core
import examples.relativity.twistors.core
import examples.relativity.twistors.scenarios
instantiate('examples.estimation.odometry.core', PGA2D)
instantiate('examples.estimation.multiview.core', PGA2D)
instantiate('examples.mechanics.tennis_racket.core', Algebra.from_pqr(3, 0, 0))
instantiate('examples.quadrics.elliptic_physics.core', Spherical3D)
print([m for m in sys.modules if m.split('.')[0] in ('matplotlib', 'PIL')])
"""


def test_mathematics_does_not_import_plotting():
    out = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", out.stdout
