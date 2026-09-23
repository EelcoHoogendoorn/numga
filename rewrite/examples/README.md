# Examples

| Topic | Examples |
| --- | --- |
| Geometry | [projection](geometry/projection/), [scenegraph](geometry/scenegraph/), [fitting](geometry/fitting/), [registration](geometry/registration/), [quadric error metrics (QEM)](geometry/qem/), [epipolar geometry & 3D reconstruction](geometry/epipolar/), [multi-view cone reconstruction & bundle adjustment](geometry/multiview/), [camera pose by gradient descent (JAX)](geometry/camera_fit.py), [Kalman pose filter](geometry/kalman/), [skinning](geometry/skinning/) |
| Quadrics | [Projective quadrics](quadrics/quadrics/), [Gaussian fit and 1σ quadric](quadrics/gaussian/), [spherical quadrics](quadrics/spherical_quadrics/), [CGA spherical quadrics](quadrics/cga_spherical_quadrics/), [sphere rendering](quadrics/conformal_elliptical/), [collision](quadrics/quadric_collision/), [Cayley–Klein geometry](quadrics/cayley_klein/), [dynamics on the 2- and 3-sphere](quadrics/elliptic_physics/), [rendering on the 3-sphere](quadrics/s3_raytracer/) |
| Mechanics | [Inertia](mechanics/inertia.py), [simplex inertia](mechanics/simplex.py), [stiffness and normal modes](mechanics/modes/), [tennis racket instability](mechanics/tennis_racket/), [Lie integrators](mechanics/lie_integrators.py), [rigid body chains (XPBD)](mechanics/xpbd/), [robot arm kinematics](mechanics/robot_arm/) |
| Relativity | [Distributed impulses, the ladder paradox and Bell's spaceships](relativity/impulse/), [curvature and gravitational waves](relativity/curvature/) |
| Electromagnetism | [Maxwell maps](electromagnetism/maxwell.py), [constitutive maps](electromagnetism/constitutive/) |
| Optics | [Thin lenses as maps on lines](optics/thin_lens/), [zoom camera with depth of field](optics/lens_camera/) |
| Symmetry | [Heat conduction, a flywheel and a crystal lattice](mechanics/symmetry/) |
| Conformal geometry | [Ray tracing Dupin cyclides on the 3-sphere](geometry/cyclides/) |
| Benchmarks | [Motor map timings](../benchmarks/motor_map.py) |

Run an example from `rewrite/` with `PYTHONPATH=src:. python -m examples.<topic>.<name>.scenarios`.

Notebooks sit next to their examples. Open one in Jupyter from inside `rewrite/`, where its first cell puts `src` on the path, or use the Open in Colab badge at its top.
