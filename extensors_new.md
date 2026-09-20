# GOAL: PRIMER ON EXTENSORS AS ACUTALLY TAKEN SHAPE IN THE NUMGA REWRITE.
by contrast with extensors.md, there is much more concrete syntax and examples to show; and some syntax details there are likely to be stale.
extensors should be mentioned in the general readme; it is likely the biggest discriminator of numga as a package; but there is more to say than can be crammed into a readme.
possibly there is more to say than can be said in a single document... the target audience is an eclectic mix. what speaks to a programmer may not speak to a mathematician; etc. my core audience is really people who want clean math expressed cleanly in code...

ive gotten pushback on showing people the extensor_cheatsheet in isolation; they feel it lacks context; which it does thats the point of a cheatsheet... but i think the structure this doc should take should take elements from the cheatsheet; lead with an elegant bit of cheatsheet type math in isolation; show a compelling output picture for the visually inclined, immeidately link to the runnable code it relates to. then dive into a section of prose, explaining what that example demonstrates.

Scope: one main narrative for people who want clean geometry expressed in code, introducing the domain knowledge each example needs. Keep implementation mechanics outside this primer; a brief explanation that binding and composition are array contractions under the hood is sufficient.

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
* the same geometric construction can span dimensions and signatures; the algebra supplies the signs, component counts, and output spaces.
* binding order and execution strategy are independent choices; binding and composition become array contractions, whose execution belongs to the backend.
* beyond outermorphisms; GA literature focuses on transformations of space, but physics and shapes are derivations, polarities, and general forms.

## Tentative example sections

Keep the introduction short and go straight into motivated examples. Each section starts with a visual, a short code blurb, and the runnable-example link, followed by the explanation. These six candidates are ordered to build on one another.

### 1. A shadow is a map

* **Example:** [projection.py](rewrite/examples/geometry/projection.py).
* **Visual:** a light, an object, the ground, and its projected shadow.
* **Blurb:** new; `shadow = (light & Point) ^ ground`, followed by `shadow(vertices)`.
* **Argument:** an ordinary geometric expression becomes a map by leaving an argument open. Its matrix follows from the geometry. Leaving the light open instead asks a different question with the same construction.

Start with the shadow portion of the example; save its epipolar geometry for further reading.

### 2. Assemble a camera before supplying its subjects

* **Example:** [lens_camera.py](rewrite/examples/sketches/lens_camera.py), with [scenegraph.py](rewrite/examples/geometry/scenegraph.py) as the composition extension.
* **Visual:** two lenses and their ray paths alongside the image as focus changes.
* **Blurb:** the cheatsheet's “Two lenses → a camera”: construct the front lens, motor it into the rear lens, compose the train, and meet its output with the sensor.
* **Argument:** maps participate in GA expressions, move with their objects, and compose before their remaining arguments arrive. Nested poses, anisotropic scaling, lenses, and projection can become one map before vertices are supplied.

### 3. Let the springs tell us how the body moves

* **Example:** [stiffness.py](rewrite/examples/mechanics/stiffness.py).
* **Visual:** the slide/bounce/rock animation, followed by the coupled motions when an angled spring is added.
* **Blurb:** the cheatsheet's “Mass points → inertia → response” leading into “Spring geometry → vibration modes”, ending with `(Twist & stiffness).eigh(Twist & inertia)`.
* **Argument:** batch and sum local responses into reusable maps. Mass distribution and spring geometry belong to setup; motion is supplied later. Pairing the response maps with another open motion constructs energy forms whose eigenvectors are physical motions.

### 4. Find the rotation that aligns two clouds

* **Example:** [registration.py](rewrite/examples/geometry/registration.py).
* **Visual:** corresponding points before and after alignment.
* **Blurb:** the cheatsheet's “Point correspondences → best rotation”: an open rotor sandwich, summed alignment, and the maximizing eigenmode.
* **Argument:** multiple open arguments construct a quadratic objective. The geometric expression generates the estimation matrix, and the eigensolver returns geometric objects. Sample batches and open rotor slots remain separate throughout.

### 5. A surface can itself be a map

* **Example:** [quadrics.py](rewrite/examples/quadrics/quadrics.py).
* **Visual:** an ellipsoid, a tangent plane, and their contact point moving together.
* **Blurb:** new; construct a dual quadric, obtain `contact = quadric(tangent_plane)`, invert its polarity, and move the whole map with a motor.
* **Argument:** extensors represent shapes as well as transformations. Input and output can occupy different geometric spaces. Inversion and composition have direct geometric meanings: a tangent plane maps to its contact point, and the inverse polarity maps the contact point back to its tangent plane.

A [CGA vortex animation](rewrite/examples/sketches/cga_quadric.py) could provide the closing illustration once polarity and transforming a surface are understood. Keep the raytracer outside the article's explanation.

### 6. What does an observer measure from curvature?

* **Example:** [curvature.py](rewrite/examples/relativity/curvature.py).
* **Visual:** a ring stretching under a gravitational wave.
* **Blurb:** the cheatsheet's “Gravitational wave → tidal acceleration”, emphasizing `tidal = curvature(observer.wedge(Vector)).commutator(observer)` and `acceleration = tidal(separation)`.
* **Argument:** binding physical context constructs another useful map. Bivector geometry becomes an observable vector response through ordinary GA operations. Introduce the observer and the measured relative acceleration before discussing the curvature operator's spectral properties.

The existing blurbs are collected in [extensor_cheatsheet.md](extensor_cheatsheet.md); shadows and quadric polarity need their own short blurbs for the primer.

## Mathematical hooks

Extensors are established multilinear algebra. The mathematical interest here is in the constructions Numga makes easy to express, connect, and investigate. These are possible threads for the primer, grounded in existing examples.

### Second moments, inertia, and exterior powers

The [Gaussian](rewrite/examples/quadrics/gaussian.py) and [inertia](rewrite/examples/mechanics/inertia.py) examples start from the same second-moment information. In a positive Euclidean space, a symmetric second-moment map `C` induces a bivector map:

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

The [symmetry example](rewrite/examples/sketches/symmetry.py) averages a map under conjugation by a finite rotor group:

```python
invariant = (group >> Vector)(tensor(group << Vector)).mean(axis=0)
```

This is a Reynolds projection onto the invariant maps. Which components survive axial symmetry, which survive cubic symmetry, and why? Replacing vectors with bivectors changes the representation while preserving the construction. This gives a concrete entry into invariant theory, with anisotropic tensors as visible examples. See [Reynolds averaging](https://cs.uwaterloo.ca/~r5olivei/courses/2021-winter-cs487/lecture16.pdf).

### How much geometry can one quadric determine?

The [Cayley–Klein example](rewrite/examples/quadrics/cayley_klein.py) starts with projective incidence and chooses a conic as an absolute. From it come distances, perpendiculars, reflections, and circles.

An extensor represents the polarity from points to lines; its inverse sends lines back to points. Projective duality becomes something directly composable in code. This offers a visually accessible mathematical story with little physics background required.

### A nonzero self-adjoint map whose eigenvalues all vanish

The [curvature example](rewrite/examples/relativity/curvature.py) constructs a nonzero curvature operator on bivectors with `R(R) = 0`. It is self-adjoint with respect to an indefinite pairing, where nilpotence and self-adjointness can coexist.

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
