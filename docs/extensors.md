# Definition

In mathematical terms, an extensor is a multi-linear map from multivectors to a multivector.

In programming terms, extensors allow one to leave open arguments to an expression, and bind them at a later time.

When doing mathematics on the blackboard, one often switches between expressions involving a specific vector, or expressions over the entire space of vectors. Extensor syntax brings that same flexibility to geometric algebra in code, combining expressivity with efficiency of the underlying code.

This is the definition of Hestenes and Sobczyk [[ca-to-gc](#ref-ca-to-gc)], for whom extensors are what tensors become when their arguments are multivectors instead of vectors. Numga's extensors are built from the products of the algebra, so they transform as the geometry they are built from does.

# Motivation

Geometric relationships deserve to be first-class objects alongside the objects they relate. Inertia, stiffness, and material responses are maps that we need to construct, combine, transform, and solve with. Extensors make those relationships part of the geometric algebra library, expressed through the same operations as the geometry that defines them.

For readers coming from linear algebra: wherever a geometric computation would use a matrix, numga has an extensor that knows what it maps from and to. The examples below use rotor sandwiches, non-uniform scalings, lenses and projections, and they all compose, invert and apply the same way. Solving, eigen- and singular-value decompositions and least squares work on them directly and return geometry: the vibration modes of example 2 come out as twists, not as columns of numbers.

For readers coming from tensor algebra: an extensor is a tensor whose slots are labelled by the kind of multivector they take, not by upper and lower index positions. The metric is part of the algebra's products, so there is nothing to raise or lower. Turning a map into a form means applying the inner product, and you write that once, as an open slot. More slots come from leaving more arguments open, not from tensor products. Index gymnastics become slot bookkeeping, and the types do the bookkeeping.

# Companion documents

* [`extensor_syntax.md`](extensor_syntax.md): the syntax and the extension methods, as a reference sheet.
* [`extensor_advanced.md`](extensor_advanced.md): maps against forms, the pairing that replaces the transpose, traces, norms and gauges.
* [`internals.md`](internals.md): what the library builds from an expression, and what runs when values are supplied.

# Examples

Capitalized names are multivector spaces and lower case names are concrete multivectors: `v ^ V` is the wedge product of a specific vector with the space of all vectors, an extensor with bivector output and one open argument.

### Index
1. [**Computer Graphics & Optics (PGA3D)**](#1-scenegraph-forward-kinematics--camera-optics-pga3d): a robot arm, two lenses and a sensor, composed into one map from each part to the screen.
2. [**Mechanics & Vibrations (PGA2D)**](#2-rigid-body-normal-modes--vibration-pga2d): stiffness and inertia as sums of maps, and vibration modes that come out as motions.
3. [**Multi-View Vision & Camera Alignment (PGA2D)**](#3-multi-view-scene-reconstruction--camera-alignment-pga2d): pixel uncertainty carried back into cones of sight, fused by addition, and cameras aligned by solving with them.
4. [**Electromagnetism & Spacetime Physics (STA)**](#4-spacetime-constitutive-relations-dispersion--relativistic-fresnel-drag-sta): materials as maps on fields, moved at relativistic speeds, and wave speeds from an SVD.
5. [**Gravitational Waves & Tidal Forces (STA)**](#5-gravitational-wave-curvature--tidal-forces-sta): a gravitational wave's curvature as a map on planes, and the tides an observer feels.
6. [**Rigid Bodies on the Sphere (Spherical3D)**](#6-rigid-bodies-on-the-sphere-spherical3d): shapes as quadric forms, drawn, collided and bounced.
7. [**Dupin Cyclides & Vortices on the 3-Sphere (Conformal Model)**](#7-dupin-cyclides--vortices-on-the-3-sphere-conformal-model): ray tracing with open forms, and shapes made by moving maps.

---

## 1. Scenegraph, Forward Kinematics & Camera Optics (PGA3D)

**Notebook**: [`examples/geometry/scenegraph/scenegraph.ipynb`](../examples/geometry/scenegraph/scenegraph.ipynb)

![Scenegraph 3D scene and 2D sensor photograph](../plots/scenegraph.png)

```python
body = pose >> scale                                     # [5] Point <- Point: each part, scaled and placed
camera = to_sensor(rear_lens(front_lens(rays)))          # [] Point <- Point: two lenses and a sensor
local_to_pixel = viewport(camera(camera_pose << body))   # [5] Point <- Point
pixels = local_to_pixel[:, None](corners[None, :])       # [5, 8] Point
```

* **One map from part to pixel.** Scaling, rigid motion, refraction and perspective compose into a single `Point <- Point` per part before any point is touched. In a graphics pipeline this is the product of the model, view and projection matrices.
* **Motors move maps as they move points.** `camera_pose << body` brings a whole map into the camera's frame, the same way `camera_pose << p` brings in a point.

---

## 2. Rigid-Body Normal Modes & Vibration (PGA2D)

**Notebook**: [`examples/mechanics/modes/modes.ipynb`](../examples/mechanics/modes/modes.ipynb)

![The three vibration modes of the coupled suspension](../plots/modes.png)

```python
stiffness = (springs * (springs & Twist) * constants).sum()   # [] Forque <- Twist
inertia = (points & points.commutator(Twist) * masses).sum()  # [] Forque <- Twist
values, modes = (Twist & stiffness).eigh(Twist & inertia)     # modes: [3] Twist
```

* **Stiffness and inertia are sums.** Each spring and each mass point adds one term. No origin is chosen, and no parallel-axis shift is needed.
* **The modes are motions.** The eigensolve runs on the two energy forms and returns twists, the motions the body vibrates in.

---

## 3. Multi-View Scene Reconstruction & Camera Alignment (PGA2D)

**Notebook**: [`examples/geometry/multiview/multiview_reconstruction.ipynb`](../examples/geometry/multiview/multiview_reconstruction.ipynb)

![Multi-view reconstruction, sight cones, splats, and pose covariance](../plots/multiview_reconstruction.png)

```python
on_planes = (Plane & Point).solve(Plane & projection)             # [] Plane <- Plane, induced by the camera
cones = on_planes(sensor_discs(projection))                       # [n_points, n_cams] Plane <- Point
splats = (poses >> cones(poses << Point)).sum(axis=-1)            # [n_points] Plane <- Point
points = (splats + w * (w & Point)).solve(w)                      # [n_points] Point

motion = -Twist.commutator(poses << points[:, None])              # [n_points, n_cams] Point <- Twist
curvature = (cones(motion) & motion).sum(axis=0)                  # [n_cams] Scalar <- (Twist, Twist)
step = curvature.solve(-gradient)                                 # [n_cams] Twist
```

* **Uncertainty travels as a shape.** Each measurement is a quadratic cost on the sensor, and the camera carries it back into a cone of sight that widens with depth.
* **Combining views is addition.** The cones of all cameras sum into a splat, a confidence ellipsoid around each scene point, and one solve finds its centre.
* **Camera alignment in pure geometry.** How a point moves under an open camera step is one commutator; joined with the cones it gives curvature and gradient, and one solve gives the Gauss-Newton step.

---

## 4. Spacetime Constitutive Relations, Dispersion & Relativistic Fresnel Drag (STA)

**Notebook**: [`examples/electromagnetism/constitutive/constitutive.ipynb`](../examples/electromagnetism/constitutive/constitutive.ipynb)

![Plane waves in isotropic glass and in a birefringent crystal](../plots/constitutive.png)

```python
electric = Bivector.commutator(t).wedge(t)               # [] Bivector <- Bivector: what observer t calls electric
glass = eps * electric + (Bivector - electric) / mu      # [] Bivector <- Bivector
moving = boost >> glass(boost << Bivector)               # [n_betas] Bivector <- Bivector: the glass, moving
wave = k.commutator(moving(k.wedge(Spatial)))            # [n_speeds, n_betas] Vector <- Spatial
wave.svdvals()                                           # near zero where light can travel
```

* **An observer is a map.** With the field left open, one line gives the part of any field that observer `t` calls electric, and glass is two such parts, weighted and added.
* **Moving a material moves a map.** A boost moves the glass the same way it moves a vector. Fresnel drag follows, without transformation rules for ε and μ.
* **Wave speeds from an SVD.** Leaving the polarization open turns Maxwell's equations for a trial wave into a map. Its singular values, over a batch of trial speeds, show which speeds light can travel at, and with which polarization.

---

## 5. Gravitational Wave Curvature & Tidal Forces (STA)

**Notebook**: [`examples/relativity/curvature/curvature.ipynb`](../examples/relativity/curvature/curvature.ipynb)

![Bead ring response to plus, cross and circular gravitational wave packets](../plots/curvature.png)

```python
nx, ny = k.wedge(x), k.wedge(y)                                        # [] Bivector: two planes along the wave
plus = nx * (nx | Bivector) - ny * (ny | Bivector)                     # [] Bivector <- Bivector
cross = eighth_turn >> plus(eighth_turn << Bivector)                   # [] Bivector <- Bivector
ricci = Vector.commutator(plus(Vector.wedge(Vector))).trace(slot=1)    # [] Scalar <- (Vector, Vector): zero
tidal = plus(t.wedge(Vector)).commutator(t)                            # [] Vector <- Vector
```

* **Curvature is a map on planes.** A gravitational wave's curvature is two dyads, and its second polarization is the same map turned by an eighth turn.
* **The textbook quantities are one line each.** The Ricci form is a trace, zero because the wave travels through vacuum. What an observer feels is the curvature with their velocity bound in: a map that stretches a ring of beads one way and squeezes it the other.

---

## 6. Rigid Bodies on the Sphere (Spherical3D)

**Notebook**: [`examples/quadrics/elliptic_physics/s2_physics.ipynb`](../examples/quadrics/elliptic_physics/s2_physics.ipynb)

![Seven ellipses spinning and colliding on the 2-sphere](../plots/spherical_quadric_physics.gif)

```python
inside = (pixels & shape(pixels)) < 0                             # drawing: one test per pixel
margin, deepest = overlap(shape, relative >> other(relative << Point))   # the other shape, in this one's frame
wrench = shape(deepest) | shape.inverse()(shape(deepest))         # push along the contact normal
impulse = -2 * closing / (wrench & inertia.inverse()(wrench))     # one body's share; the pair adds both
```

* **Shapes as quadric forms.** Drawing, collision and contact all come from the form: a pixel is inside where it is negative, and two shapes are apart exactly when some blend of their forms is positive.
* **One formula for the bounce.** The impulse follows from the contact wrench and each body's inverse inertia, and energy and momentum are conserved.

---

## 7. Dupin Cyclides & Vortices on the 3-Sphere (Conformal Model)

**Notebook**: [`examples/geometry/cyclides/cyclides.ipynb`](../examples/geometry/cyclides/cyclides.ipynb)

![A cone-tipped cyclide carried around a vortex circle, linked with a ring on that circle](../plots/cyclides_linked_vortex.gif)

```python
form = Point & surfaces                      # [n] Scalar <- (Point, Point): zero on the surface
ray_bend                                     # [] Point <- (Direction, Direction): how a ray bends
quartic = form(ray_bend, ray_bend)           # [n] Scalar <- (Direction, Direction, Direction, Direction)

tori = dilation >> tubes(dilation << Point)  # [3] Sphere <- Point: dilated tubes are tori
rolled = flow >> tori(flow << Point)         # [36] Sphere <- Point: carried around a vortex
```

* **A form with four open slots.** Feeding the ray's bend into both slots of the surface's form leaves four open directions: the leading coefficient of every pixel's quartic, from one binding.
* **Shapes are made by moving maps.** A dilation bends tubes into tori and Dupin cyclides, and a circle's exponential carries a surface around it.


# References

* <a id="ref-ca-to-gc"></a>**[ca-to-gc]** D. Hestenes and G. Sobczyk, *Clifford Algebra to Geometric Calculus: A Unified Language for Mathematics and Physics*, Reidel, 1984. Extensors are defined in Section 3-10, "Tensors"; extensor fields and their differentials in Section 4-1. [Link](https://math.mit.edu/~dunkel/Teach/18.S996_2022S/books/Hestenes-Sobczyk1984_Book_CliffordAlgebraToGeometricCalc.pdf)
