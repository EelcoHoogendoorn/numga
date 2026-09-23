# GOAL: PRIMER ON EXTENSORS AS ACUTALLY TAKEN SHAPE IN THE NUMGA REWRITE.
by contrast with extensors.md, there is much more concrete syntax and examples to show; and some syntax details there are likely to be stale.
extensors should be mentioned in the general readme; it is likely the biggest discriminator of numga as a package; but there is more to say than can be crammed into a readme.
possibly there is more to say than can be said in a single document... the target audience is an eclectic mix. what speaks to a programmer may not speak to a mathematician; etc. my core audience is really people who want clean math expressed cleanly in code...

ive gotten pushback on showing people the extensor_cheatsheet in isolation; they feel it lacks context; which it does thats the point of a cheatsheet... but i think the structure this doc should take should take elements from the cheatsheet; lead with an elegant bit of cheatsheet type math in isolation; show a compelling output picture for the visually inclined, immeidately link to the runnable code it relates to. then dive into a section of prose, explaining what that example demonstrates.

Scope: one main narrative for people who want clean geometry expressed in code, introducing the domain knowledge each example needs. Keep implementation mechanics outside this primer; a brief explanation that binding and composition are array contractions under the hood is sufficient.





# list of common maps and their extensor equivalent
 * rotation matrix;                         Point <- Point, Line <- Line, etc
 * perspective projection matrix;           Point <- Point, Line <- Line, etc
 * inertia map;                             AntiBivector <- Bivector
 * stiffness matrix                         AntiBivector <- Bivector
 * second moment quadric form;              Point <- Plane
 * Electromagnetic constitutive tensor;     Bivector <- Bivector
 * Stress–energy tensor                     Vector <- Vector
 * Pose covariance / Kalman gain            Bivector <- Bivector
 * perspective cone / precision quadric;    Plane <- Point
 * Gauss-Newton normal equations;          Twist <- Twist

Note that being such unary maps we may also encounter higher arity extensors

 * Kinetic or elastic energy form	        Scalar <- Bivector, Bivector
 * Crystal elasticity	                    Scalar <- Vector, Vector, Vector, Vector
 * Camera with an open pupil position	    Point <- Point, Point
 * least square rotor fit                   Scalar <- Rotor, Rotor

note that many of these have a natural grading in GA, beyond their formulation in matrix/tensor conventions, that operate on grade 1 vectors only

## aspects to showcase

 1 **Construct maps directly from geometry.** Leave an argument open in an ordinary GA expression: a shadow is join-then-meet, and its projection matrix follows from that expression.
 2 **Choose what remains open, and when to bind it.** A shadow can depend on the object point or on the light position. Inertia binds the mass distribution at setup and velocity during simulation.
 3 **Use GA operations on maps themselves.** Reverse, pair, wedge, and otherwise combine maps in the same expressions as multivectors; `residual.reverse() | residual` constructs a fitting objective.
 4 **Go beyond unary maps.** Multiple open arguments describe energy forms, stereo correspondence, a camera with an open pupil, and crystal elasticity. Binding some arguments produces another useful map.
 5 **Transform whole geometric relationships.** Carry inertia between body and world frames, view a medium from a moving observer, or pull a quadric back through a camera. Account for the map's inputs as well as its output.
 6 **Compose unlike transformations and choose the evaluation order.** Rotations, nonuniform scaling, lenses, and projection can become one map before a vertex batch is supplied. This connects the geometric construction to the usual efficiency argument for scene graphs.
 7 **Apply linear algebra without losing geometric types.** Invert a polarity, solve for motion from momentum, or extract fitted planes and vibration modes from eigenproblems.
 8 **Represent more than transformations of space.** Points, lines, planes, velocities, and momenta give maps meaningful input/output signatures. A quadric is a shape represented by a map; inertia and constitutive tensors are physical responses.
 9 **Assemble responses from contributions.** Weight and sum particle inertias, spring stiffnesses, or point moments; average transformed material responses over a symmetry group. Sample batches and open slots are independent axes of the construction.

## one line takeaways

* no need to choose between GA/quats versus matrices. extensors subsume matrices
* binding point of view; open extensor slots are late bound arguments (some details; lifting of expressions etc)
* viewed mathematically; operations on concrete multivectors; versus operations over the abstract space of all multivectors
* control over order of operations `do(stuff(to(vertices)))` vs `do(stuff(to()))(vertices)`; scenegraph example
* code organisation; inerta in a physics sim requires binding the mass distribution at setup time, but monemtum binding in the sim loop. no way to restructure the code to make that go away
* linear abgebraic operations as typed operations with the GA-algebra; inverse/solve/eigs returning a batch of planes, etc.
* GA operations work on maps and forms too; `residual.reverse() | residual` constructs a misfit form using the same operations as concrete geometry.
* higher arity gives tensors beyond matrices; multiple open arguments describe cross products, energy forms, and rotor-fitting objectives.
* the geometry generates the coefficients; the rotor-fitting matrix emerges from an open sandwich, and its coefficient array represents the expression we wrote.
* collections and open arguments are separate concepts; broadcast, weight, sum, and average batches of maps while preserving each slot's algebra and subspace.
* geometric responses move with their objects; `motor >> inertia(motor << Bivector)` transforms the whole response, accounting for both its input and output.
* binding order and execution strategy are independent choices; binding and composition become array contractions, whose execution belongs to the backend.
* beyond outermorphisms; GA literature focuses on transformations of space, but physics and shapes are derivations, polarities, and general forms.

## Tentative example sections

Keep the introduction short and go straight into motivated examples. Each section starts with a visual, a short code blurb, and the runnable-example link, followed by the explanation. These ten candidates offer different routes through the material.

### 1. A shadow is a map

* **Example:** [geometry/projection/](rewrite/examples/geometry/projection/).
* **Aspects:** 1, 2, 3.
* **Core LOC:** 4 (2 for a single shadow).
* **Visual:** a light, an object, the ground, and its projected shadow.
* **Blurb:** new; `shadow = (light & Point) ^ ground`, followed by `shadow(vertices)`.
* **Core:**
  ```python
  # Join point with light into ray, meet with ground plane
  shadow = (light & Point) ^ ground
  projected = shadow(vertices)

  # Reopen the light slot: sweep one corner under moving sunlight
  trail = ((Point & corner) ^ ground)(sun_path)
  ```
* **Argument:** an ordinary geometric expression becomes a map by leaving an argument open. Its matrix follows from the geometry. Leaving the light open instead asks a different question with the same construction.

Start with the shadow portion of the example; save its epipolar geometry for further reading.

### 2. Assemble a camera before supplying its subjects

* **Example:** [optics/lens_camera/](rewrite/examples/optics/lens_camera/), with [geometry/scenegraph/](rewrite/examples/geometry/scenegraph/) as the composition extension.
* **Aspects:** 1, 2, 3, 4, 5, 6, 7, 8.
* **Core LOC:** 22 (5 for the two-lens cheatsheet core).
* **Visual:** two lenses and their ray paths alongside the image as focus changes.
* **Blurb:** the cheatsheet's “Two lenses → a camera”: construct the front lens, motor it into the rear lens, compose the train, and meet its output with the sensor.
* **Core:**
  ```python
  front = Line - (origin & (Line ^ lens_plane)) / focal_length
  rear = placement >> front(placement << Line)
  train = rear(front)

  camera = train(Point & Point) ^ sensor_plane
  image = camera(subject, aperture_point)
  ```
* **Argument:** maps participate in GA expressions, move with their objects, and compose before their remaining arguments arrive. Nested poses, anisotropic scaling, lenses, and projection can become one map before vertices are supplied.

### 3. Let the springs tell us how the body moves

* **Example:** [mechanics/modes/](rewrite/examples/mechanics/modes/).
* **Aspects:** 1, 2, 3, 4, 7, 8, 9.
* **Core LOC:** 8 (5 through the eigenproblem).
* **Visual:** the slide/bounce/rock animation, followed by the coupled motions when an angled spring is added.
* **Blurb:** the cheatsheet's “Mass points → inertia → response” leading into “Spring geometry → vibration modes”, ending with `(Twist & stiffness).eigh(Twist & inertia)`.
* **Core:**
  ```python
  lines = (anchors & attachments).normalized()
  extension = Twist & lines
  stiffness = (lines * extension * spring_constants).sum()

  inertia = (mass_points & mass_points.commutator(Twist) * masses).sum()
  values, modes = (Twist & stiffness).eigh(Twist & inertia)
  ```
* **Argument:** batch and sum local responses into reusable maps. Mass distribution and spring geometry belong to setup; motion is supplied later. Pairing the response maps with another open motion constructs energy forms whose eigenvectors are physical motions.

### 4. Find the rotation that aligns two clouds

* **Example:** [geometry/registration/](rewrite/examples/geometry/registration/).
* **Aspects:** 1, 3, 4, 7, 8, 9.
* **Core LOC:** 10 (3 for the rotation solve alone).
* **Visual:** corresponding points before and after alignment.
* **Blurb:** the cheatsheet's “Point correspondences → best rotation”: an open rotor sandwich, summed alignment, and the maximizing eigenmode.
* **Core:**
  ```python
  rotated = Rotor.sandwich(source)
  alignment = target.scalar_product(rotated).sum()

  values, rotors = ((alignment + alignment.transpose()) * 0.5).eigh()
  rotation = rotors[-1].normalized()
  ```
* **Argument:** multiple open arguments construct a quadratic objective. The geometric expression generates the estimation matrix, and the eigensolver returns geometric objects. Sample batches and open rotor slots remain separate throughout.

### 5. A surface can itself be a map

* **Example:** [quadrics/quadrics/](rewrite/examples/quadrics/quadrics/).
* **Aspects:** 1, 3, 5, 7, 8, 9.
* **Core LOC:** 13, including support-plane construction.
* **Visual:** an ellipsoid, a tangent plane, and their contact point moving together.
* **Blurb:** new; construct a dual quadric, obtain `contact = quadric(tangent_plane)`, invert its polarity, and move the whole map with a motor.
* **Core:**
  ```python
  # Dual quadric maps tangent planes to contact points; inverse maps back
  dual = (axes * (Plane & axes)).sum(axis=0) - center * (Plane & center)
  contact = dual(tangent_plane)

  primal = dual.inverse()
  tangent = primal(contact)

  # Moving the shape transforms the whole map
  moved_dual = motor >> dual(motor << Plane)
  ```
* **Argument:** extensors represent shapes as well as transformations. Input and output can occupy different geometric spaces. Inversion and composition have direct geometric meanings: a tangent plane maps to its contact point, and the inverse polarity maps the contact point back to its tangent plane.

A [CGA vortex animation](rewrite/examples/sketches/cga_quadric.py) could provide the closing illustration once polarity and transforming a surface are understood. Keep the raytracer outside the article's explanation.

### 6. What does an observer measure from curvature?

* **Example:** [relativity/curvature/](rewrite/examples/relativity/curvature/).
* **Aspects:** 1, 2, 3, 5, 6, 8, 9.
* **Core LOC:** 11.
* **Visual:** a ring stretching under a gravitational wave.
* **Blurb:** the cheatsheet's “Gravitational wave → tidal acceleration”, emphasizing `tidal = curvature(observer.wedge(Vector)).commutator(observer)` and `acceleration = tidal(separation)`.
* **Core:**
  ```python
  ribbon_x = wavevector.wedge(x)
  ribbon_y = wavevector.wedge(y)
  curvature = amplitude * (
      ribbon_x * (ribbon_x | Bivector) - ribbon_y * (ribbon_y | Bivector)
  )

  tidal = curvature(observer.wedge(Vector)).commutator(observer)
  acceleration = tidal(separation)
  ```
* **Argument:** binding physical context constructs another useful map. Bivector geometry becomes an observable vector response through ordinary GA operations. Introduce the observer and the measured relative acceleration before discussing the curvature operator's spectral properties.

### 7. Depth of field from an aperture cone

* **Example:** the aperture and cone construction in [optics/lens_camera/](rewrite/examples/optics/lens_camera/).
* **Aspects:** 1, 2, 3, 5, 6, 7, 8.
* **Core LOC:** 7 through image-cone construction, given the composed lens map; 4 if the posed aperture quadric is also supplied. Sensor sampling and rasterization are excluded.
* **Visual:** points at different depths becoming sharp or blurred as the aperture, focus, and sensor tilt change.
* **Blurb:** new; construct and place the aperture quadric, project from each subject onto its plane, and pull the quadric back to obtain the subject's ray cone. Pull that cone through `collineation.inverse()` to obtain the image cone; its section on the sensor is the blur conic.
* **Core:**
  ```python
  # Project subject through aperture plane, pull back aperture ball into ray cone
  project = (subject & Point) ^ aperture_plane
  cone = project.transpose()(Plane.dual()).dual_inverse()(pupil_ball(project))

  # Pull cone back through inverse lens train to obtain image blur cone
  back = collineation.inverse()
  image_cone = back.transpose()(Plane.dual()).dual_inverse()(cone(back))
  ```
* **Argument:** a whole family of rays becomes one geometric object. Composing a quadric with projection constructs a cone; transporting it through the lens train carries the whole family at once. This connects maps as shapes, input/output transformations, composition, and inversion to the visible physics of depth of field.

### 8. Track a pose and its uncertainty

* **Example:** [geometry/kalman/](rewrite/examples/geometry/kalman/).
* **Aspects:** 1, 2, 3, 5, 7, 8.
* **Core LOC:** 10: 7 for prediction and measurement correction, plus 3 to map pose covariance to the position ellipses.
* **Visual:** the true path, drifting dead reckoning, noisy measurements, and the filtered path with its uncertainty ellipses.
* **Blurb:** new; `adjoint = step << Bivector` transports uncertainty during prediction, `gain = sigma((sigma + R).inverse())` weights the full-pose measurement innovation, and exponentiating that weighted innovation updates the motor. A commutator supplies the position Jacobian for the uncertainty ellipse.
* **Core:**
  ```python
  # Predict: step adjoint transports pose covariance
  adjoint = step << Bivector
  sigma = adjoint(sigma(adjoint.transpose())) + process_noise
  estimate = estimate * step

  # Update: invert sum of covariances to weight Lie-algebra innovation
  gain = sigma((sigma + measurement_noise).inverse())
  innovation = (estimate.inverse() * measurement).log() * 2
  estimate = estimate * (gain(innovation) * 0.5).exp()
  sigma = sigma - gain(sigma)
  ```
* **Argument:** pose, pose error, and uncertainty have different geometric roles: a motor, a bivector, and a map on bivectors. Their interaction stays in one language. Open sandwiches construct the motion Jacobian, covariance maps move with the frame and accumulate process noise, and inversion produces a gain that acts directly on the logarithm of the relative motor. The same covariance can then be pushed through a position map for display.

### 9. Mesh simplification with quadric error metrics

* **Example:** [geometry/qem/](rewrite/examples/geometry/qem/).
* **Aspects:** 1, 2, 3, 5, 7.
* **Core LOC:** 4.
* **Visual:** corner apex preservation, crease ridge cylinder, and edge collapse.
* **Blurb:** plane distance dyads $P * (P \& \text{Point})$ summed over incident faces form the Garland–Heckbert quadric error metric. Edge collapse sums quadrics across endpoints $Q_{\text{edge}} = Q_a + Q_b$, and the optimal vertex solves $Q_{\text{edge}}.\text{lstsq}(\infty).\text{normalized}()$ directly against the plane at infinity.
* **Core:**
  ```python
  # Vertex quadrics as sums of rank-1 plane dyads
  qa = (planes_a * (planes_a & Point)).sum(axis=0)
  qb = (planes_b * (planes_b & Point)).sum(axis=0)

  # Edge contraction combines quadrics by extensor addition
  q_edge = qa + qb

  # Optimal collapse vertex has vanishing spatial gradient: Q(X) ∝ mv.w
  v_edge = q_edge.lstsq(mv.w).normalized()
  ```
* **Argument:** mesh simplification is fundamentally a sum over rank-1 projection operators. Each face plane defines a rank-1 dyad; vertex quadrics are sums over incident dyads; edge collapse is the sum of vertex quadrics. The optimal position is found directly by solving the extensor against the plane at infinity, where vanishing spatial gradient enforces the minimum distance condition.

### 10. Multi-view bundle adjustment with perspective cones

* **Example:** [geometry/multiview/](rewrite/examples/geometry/multiview/).
* **Aspects:** 1, 3, 5, 6, 7, 8, 9.
* **Core LOC:** 12: 3 for landmark triangulation via the pole of infinity, plus 7 for the Lie-algebra Gauss-Newton pose update loop, plus 2 for camera sensor pullback.
* **Visual:** multi-camera convergent rig, initial vs. optimized camera frustums and landmarks, and 3D Gaussian splat precision ellipsoids.
* **Blurb:** sensor disk quadrics pull back through projective cameras into 3D perspective cone quadrics ($Plane \leftarrow Point$). Summing cones across cameras triangulates 3D landmarks as poles of the plane at infinity: `q_fused.inverse()(mv.w).normalized()`. Bundle adjustment evaluates polar plane residuals and Lie-algebra commutator Jacobians `-cones(Twist.commutator(points))` directly on the cones, accumulating the Gauss-Newton normal equations without ray-tracing or Cartesian coordinates.
* **Core:**
  ```python
  # Pull sensor uncertainty disks back into 3D perspective cone quadrics
  pullback = cameras.transpose()(Plane.dual()).dual_inverse()
  cones = pullback(sensor_quadrics(cameras))

  # Triangulate: fused precision quadric's pole of infinity is the 3D landmark
  world_cones = motors >> cones(motors << Point)
  q_fused = world_cones.sum(axis=-1)
  points = q_fused.inverse()(mv.w).normalized()

  # Bundle adjustment: polar plane residuals and se(3) commutator Jacobians
  local_points = motors << points[:, None]
  res = cones(local_points)
  j = -cones(Twist.commutator(local_points))
  h = (j.transpose()(j)).sum(axis=0)
  rhs = -(j.transpose()(res)).sum(axis=0)
  motors = motors * (h.lstsq(rhs, rcond=1e-4) * 0.5).exp()
  ```
* **Argument:** photogrammetry and bundle adjustment usually rely on hybrid representations: 2D pixel coordinates, 3D sight rays, and separate rotation/translation parameterizations. Extensors unify the entire pipeline under projective quadric geometry. The camera map pulls 2D sensor uncertainty into a true 3D cone; inverting the fused quadric evaluates directly on the plane at infinity to extract Euclidean centroids without nullspace eigenvalue squaring; and the Gauss-Newton normal equations assemble by contracting polar plane Jacobians against open se(3) twist commutators. Every step—from sensor measurement to 3D Gaussian splat covariance to camera pose optimization—operates on the same geometric types.

The existing blurbs are collected in [extensor_cheatsheet.md](extensor_cheatsheet.md); shadows, quadric polarity, aperture cones, Kalman filtering, QEM mesh simplification, and multi-view bundle adjustment need their own short blurbs for the primer.

### Ranking by core logic size

Logical LOC in the current examples: one mathematical statement per line, regardless of wrapping. Exclude imports, types, scenario inputs, comments, plotting, checks, and bookkeeping. Include geometric helpers and derived geometry used by the visual; count shared math only once across scenarios. Numerical plumbing, such as integration, counts at its call site.

| Rank | Example | Core LOC | What is counted |
| --- | --- | ---: | --- |
| 1 | Shadow | 4 | Point-light and sunlight maps, reopening the light slot, and applying them. A single shadow needs only 2. |
| 1 | QEM mesh simplification | 4 | Assemble vertex plane dyads, combine across the edge, and solve the optimal vertex against infinity. |
| 3 | Aperture-cone camera | 7 | Construct and place the aperture quadric, then build and transport the ray cones through a supplied lens map. This isolates the cone path from the full lens-camera example. |
| 4 | Stiffness / inertia | 8 | Assemble both responses, solve for modes, and derive body/attachment motion and spring extension. The construction through the eigenproblem is 5. |
| 5 | Registration | 10 | Center the clouds, solve the sandwich alignment, and recover translation. The rotation solve alone is 3; the alternative one-sided fit is excluded here. |
| 5 | Kalman filtering | 10 | Predict and correct pose and covariance, then derive the position-uncertainty eigenmodes. The filter loop alone is 7. |
| 7 | Curvature | 11 | Construct and combine polarizations, bind the observer, evaluate acceleration, and integrate the detector response. |
| 8 | Multiview bundle adjustment | 12 | Pull back sensor uncertainty into 3D cones, triangulate landmarks via the pole of infinity, and run Lie-algebra Gauss-Newton pose updates. The pose update loop alone is 7; landmark triangulation is 3. |
| 9 | Quadric polarity | 13 | Construct the quadric, find its support plane, map between tangent and contact, and transport the map and contact geometry. Includes the 5-statement support-plane helper. |
| 10 | Lens camera | 22 | Point and line lens maps, placement and composition, focus/sensor geometry, aperture and blur-cone pullbacks, and ray paths. The two-lens cheatsheet core is 5. |

## Mathematical hooks

Extensors are established multilinear algebra. The mathematical interest here is in the constructions Numga makes easy to express, connect, and investigate. These are possible threads for the primer, grounded in existing examples.

### Second moments, inertia, and exterior powers

The [Gaussian](rewrite/examples/quadrics/gaussian/) and [inertia](rewrite/examples/mechanics/inertia.py) examples start from the same second-moment information. In a positive Euclidean space, a symmetric second-moment map `C` induces a bivector map:

```text
J(a ∧ b) = C(a) ∧ b + a ∧ C(b)
```

This is the infinitesimal exterior-square action: the coefficient of `t` in `(a + t C(a)) ∧ (b + t C(b))`. Its eigenvalues are pairwise sums `λᵢ + λⱼ`. In three dimensions, identifying bivectors with axial vectors gives the familiar inertia map `trace(C)·identity − C`.

In bilinear-form language this connects to the Kulkarni–Nomizu product of the second-moment form with the metric, a construction of algebraic curvature tensors. A worthwhile thread to develop: the same data appearing as covariance, inertia, and an induced form on bivectors. See the [Kulkarni–Nomizu definition and curvature conventions](https://pschwahn.github.io/assets/grimoire.pdf).

For the spherical and Euclidean PGA examples (positive and null metric directions), the concrete lift from a mass-weighted `moment: Point <- Plane` to `inertia: AntiBivector <- Bivector` is:

```python
import numpy as np

ga = moment.algebra
mv = moment.context.multivector
Bivector = ga.gatype.bivector()

vectors = mv.vector(np.eye(ga.dimension))
inertia = (
    moment(vectors) & vectors.dual().commutator(Bivector)
).sum(axis=0)
```

The vector basis and its plain dual form the reciprocal plane/point bases. This contracts the second-moment columns with the point-to-momentum construction, leaving the bivector argument open. The basis is fixed by the algebra; the lift is linear in `moment`. Checked against direct point-cloud inertia in algebra dimensions 3–5 for both spherical and Euclidean PGA signatures.

### What tensors does a symmetry group permit?

The [symmetry example](rewrite/examples/mechanics/symmetry/) averages a map under conjugation by a finite rotor group:

```python
invariant = (group >> Vector)(tensor(group << Vector)).mean(axis=0)
```

This is a Reynolds projection onto the invariant maps. Which components survive axial symmetry, which survive cubic symmetry, and why? Replacing vectors with bivectors changes the representation while preserving the construction. This gives a concrete entry into invariant theory, with anisotropic tensors as visible examples. See [Reynolds averaging](https://cs.uwaterloo.ca/~r5olivei/courses/2021-winter-cs487/lecture16.pdf).

### How much geometry can one quadric determine?

The [Cayley–Klein example](rewrite/examples/quadrics/cayley_klein/) starts with projective incidence and chooses a conic as an absolute. From it come distances, perpendiculars, reflections, and circles.

An extensor represents the polarity from points to lines; its inverse sends lines back to points. Projective duality becomes something directly composable in code. This offers a visually accessible mathematical story with little physics background required.

### A nonzero self-adjoint map whose eigenvalues all vanish

The [curvature example](rewrite/examples/relativity/curvature/) constructs a nonzero curvature operator on bivectors with `R(R) = 0`. It is self-adjoint with respect to an indefinite pairing, where nilpotence and self-adjointness can coexist.

Binding an observer produces a tidal-response map with nonzero eigenvalues. That construction is not a similarity transformation: it changes the domain and the observation being made. The example connects null subspaces, metric adjoints, and observer-dependent measurements. See the [bivector classification of the Weyl operator](https://arxiv.org/abs/0909.1160).

### Identities on entire spaces

Open slots construct the coefficient tensor of a multilinear expression. Equality of those tensors establishes the identity for every input in the specified spaces.

Numga's [exact rational backend](rewrite/src/numga/backend/exact.py) lets such comparisons be exact for a fixed dimension and signature. This offers a way to explore identities, symmetries, and signature-dependent cancellations alongside numerical examples.

For the primer, symmetry and projective polarity are the strongest immediate candidates. The second-moment/exterior-power connection is a promising next development. Each supplies a mathematical question that motivates the syntax.

## Points from the chats

1. **The cheatsheet needs context.** Compact expressions work once the reader knows the problem. A GA practitioner may still lack the optics, mechanics, or relativity background needed to see what an example accomplishes. Introduce the objects and the question, show a labelled picture or result, link the runnable example, then explain what the expression demonstrates.

2. **Explain late binding through an example.** “GA with some arguments bound later” is a useful description once the reader has seen it happen. Show which arguments are supplied now, which remain open, and how an ordinary multivector expression becomes a map. Moving between a concrete bivector and the space of all bivectors should feel as natural in code as it does on the blackboard.

3. **Faithful mathematical expression is itself a contribution.** The motivation was seeing Dorst's inertia formulas and wanting to implement them generally and efficiently without losing their form. Established mathematics can still benefit from a computational language that expresses it directly. The core audience is people who want clean mathematics expressed cleanly in code.

4. **GA and matrices belong in the same language.** General linear and multilinear maps can participate in GA expressions, retain their geometric input and output types, and support composition, inversion, solves, and spectral operations. A matrix representation fits within that picture. The reader should see why there is no need to choose between GA operations and the linear algebra their application requires.

5. **Give the explicit-basis alternative a fair comparison.** In NumPy, pushing batches of basis blades through nullary expressions can construct the same maps with manageable code and arithmetic. The main burden is tracking which array axes represent samples or open arguments, together with blade layouts and broadcasting. The rotor-fitting comparison makes that bookkeeping visible: one sample axis and two independent rotor-basis axes. Avoid blanket claims that this alternative necessarily causes enormous code growth or a large performance penalty.

6. **Binding order gives control over computation.** A scenegraph can compose rotations and anisotropic scalings into one map before applying it to a large vertex batch. Sequential application is also possible, with different intermediate storage and memory traffic. Explicit basis construction can recover the composed-map strategy too; extensors make that strategy part of the typed algebra. Performance claims should follow the actual execution path and workload.

7. **Binding boundaries also express program structure.** Inertia binds a body's mass distribution during setup; velocities or momenta arrive in the simulation loop. These quantities naturally belong to different scopes and times. Small, single-scope snippets understate the value of passing the resulting typed map around a larger program instead of carrying implicit array conventions across those boundaries.

8. **A unified language can lead to new connections.** Its value includes making neighbouring subjects accessible through familiar constructions. The work on extensors led naturally into quadrics, for example. That is a reason to discuss the framework even with readers more interested in mathematics than implementation: a common language influences which ideas people explore and connect.

9. **Beyond outermorphisms: derivations, polarities, and forms.** Standard GA literature focuses almost exclusively on outermorphisms—linear maps on vectors extended to blades by preserving the wedge product, f(a ∧ b) = f(a) ∧ f(b). Outermorphisms only represent transformations of the underlying space. But the linear maps that actually describe physics and shapes are rarely outermorphisms: inertia is a derivation, quadrics are polarities mapping across grades (planes to points), and curvature is an endomorphism on bivectors. Because outermorphisms cannot express these, GA traditionally ceded general multilinear maps to index-heavy tensor calculus. Extensors bring general linear and multilinear maps into the algebra without falling back to index gymnastics.

10. **GA versus tensor algebra: a false boundary.** The divide between GA and tensor calculus is largely historical. Tensor algebra handles multilinear maps via coordinates and indices, while GA handles subspace geometry via coordinate-free products. Extensors show that Clifford algebra already contains the machinery for general multilinear maps between graded subspaces; open slots let geometric expressions compile into the exact array contractions needed. There is no need to abandon coordinate freedom to do multilinear algebra.
