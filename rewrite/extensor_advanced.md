# Extensors, advanced

Outline. Each section becomes a few paragraphs and one code block, in the voice of
[`extensors.md`](extensors.md). The syntax is in [`extensor_syntax.md`](extensor_syntax.md);
this document is about the ideas that sit behind it.

## 1. Maps and forms

- A map `B <- A` and a form `Scalar <- (A*, A)` hold the same numbers with one slot on the
  other side of the arrow. The two directions of conversion are not symmetric.
- Map to form is a pairing, cheap and always available: `Dual & map`. `Twist & stiffness` is
  the energy form of the stiffness map.
- Form to map is a solve of that pairing, `(Dual & A).solve(form)`: a form has to be asked
  which output it means.
- Two pairings are on offer. The regressive product is metric-free and puts the output in the
  dual space. The inner product `|` uses the metric and keeps the output in the same space.
  That choice is what "raising an index" means here.
- What each face is for. Maps compose, invert, and transport by sandwich. Forms add, are
  differentiated as costs, are solved against linear forms, and are the object of
  eigenproblems, including the generalized one between two forms, which needs no metric.
- Generalized eigenproblems as metric-free Rayleigh quotients: `pe_form.eigh(ke_form)` finds
  the stationary values of the energy ratio $\lambda = V(x)/T(x) = \text{pe}(x, x)/\text{ke}(x, x)$
  directly. Inverting inertia to form $M^{-1}K$ is a matrix-package artifact that destroys
  symmetry and demands an artificial coordinate metric.
- Tensor view: a map is a (1,1) tensor, a form is (0,2); numga tracks the difference by slot
  types, not by index placement.

## 2. Quadrics, the worked case

- Three faces of one quadric: polarity map `Plane <- Point`, form `Scalar <- (Point, Point)`,
  dual quadric `Point <- Plane` (the inverse polarity, when it exists).
- Conversions: `quadric & Point` gives the form; `(Plane & Point).solve(form)` gives the map.
- The plane-first convention and its reason: the regressive product of grades r and s carries
  (−1)^((n−r)(n−s)); incidence of a plane with a point is symmetric in 2D and antisymmetric
  in 3D. Keep the plane on the left, in dyads too: `normal * (normal & Point)`.
- Rule of thumb for choosing a face: maps to compose or invert, forms to sum, differentiate a
  cost, or solve.

## 3. Why there is no transpose, and what replaces it

- The coefficient transpose identifies a space with its dual by index label: a hidden
  Euclidean metric on coefficients. Right only in a Euclidean orthonormal blade basis.
- Three things hid under it, each with a geometric spelling:
  - the map on planes induced by a map on points,
    `on_planes(T) = (Plane & Point).solve(Plane & T)`, with the identity
    `on_planes(T)(l) & p == l & T(p)`, valid for singular `T`. This is the universal
    categorical adjoint: it holds for any pairing, requires no invertibility, and defines
    pullbacks through non-invertible maps (like camera projections);
  - the metric adjoint: `|` against an explicit metric form;
  - squares of residuals without $J^T W J$: when a cost is a quadric form $Q$ and kinematics
    is $J$, parameter curvature is simply $Q$ with $J$ in both slots: `Q(J) & J`. The
    transpose in $J^T W J$ was only ever the pullback of a Euclidean metric through an
    index-labeled matrix.
- The coordinate picture: the same swivel of coefficients, but the permutation and the signs
  come from the product's structure constants, which blades meet to a scalar and with what
  sign, instead of from index labels.

## 4. Inverse versus inverse

- Multivector inverse under the geometric product; map inverse under composition; form solve
  against a linear form. One word, three products, no conflict.
- A batch is not a frame: `basis.inverse()` inverts each vector, and is the reciprocal frame
  only by the accident of orthonormality. The reciprocal of a frame comes from solving the
  pairing.
- The identity map is a bare type: `Bivector - electric` is the complementary projector.

## 5. Least squares, by arity

- Nullary: coefficients of a linear combination along a batch axis.
- Unary: a map against a right-hand side, with pinv's cutoff.
- Forms: `form(x, ·) = linear(·)`; a right-hand side with leading slots yields a map.
- Tensor: solving a construction for an unknown map. The inertia-to-moment recovery is the
  example: `(Point & Plane.dual().commutator(Bivector)).lstsq(inertia)` infers the
  `Point <- Plane` slots from the types.
- Which registration dispatch picks, and from what.

## 6. Second moments and inertia

- The moment as a dual quadric: `(points * (Plane & points) * masses).sum(axis=0)`.
- Moment to inertia (verified): diagonalize `Plane & moment`; the moment's images of the
  eigenplanes are principal points carrying mass one over the eigenvalue; sum their momentum
  dyads `(principal & principal.commutator(Bivector)) / values`.
- Inertia to moment: the tensor solve of section 5.
- Why the classical route, `S = ½ tr(J) 1 − J`, is not the geometric spelling: it needs the
  Euclidean metric and a frame. What the trace of the inertia map is under the pairing
  (the mass) versus under the metric (`tr J`).

## 7. Norms without a metric

- Bulk and weight: the PGA scalar product is blind to ideal blades, the complement's scalar
  product sees exactly those, and their sum is the coefficient norm.
  `residual.reverse().scalar_product(residual)` plus the same on `residual.dual()`.
- Reading a pseudoscalar as a number through `.dual()`.
- When the coefficient norm is a legitimate choice and when it is an accident.

## 8. Traces, metrics, and the gravity example

- `trace(slot)` pairs an output with one input by matching blades and never consults the
  metric. Slots are numbered in order of appearance.
- An input-to-input contraction needs either a metric or the pairing; there is no third way.
- The Ricci form as the worked example: `Vector.commutator(R(Vector.wedge(Vector))).trace(slot=1)`.
  The inner product with an open vector is the one explicit metric; the trace does the rest;
  no frame and no reciprocal basis anywhere. Bianchi with the third vector open, as a map
  that must vanish.

## 9. Observer decompositions and spacetime lifting

- A 3D material law $\mathbf{D} = \boldsymbol{\epsilon}\mathbf{E}$ is a spatial vector map,
  `Vector <- Vector`.
- An observer's timelike 4-velocity $t$ decomposes 6D field bivectors into 3D electric and
  magnetic vectors: `Bivector.commutator(t)`.
- Passing this extractor into the 3D permittivity lifts the spatial law into a 6D spacetime
  extensor, `Bivector <- Bivector`: `permittivity(Bivector.commutator(t)).wedge(t)`.
- Moving media and relativistic Fresnel drag require no coordinate Lorentz boosts: simply
  sandwich the extensor by the boost rotor: `boost >> medium(boost << Bivector)`.
- Topological axion electrodynamics as a bare dual extensor: `mv.scalar([alpha]) * Bivector.dual()`.

## 10. Homogeneous gauge

- Quadrics on homogeneous points have a null direction along each point's own scale.
- Triangulation pins it with the dyad `w * (w & Point)`, turning the vertex into the pole of
  the plane at infinity.
- A Newton step over points must not see that direction either, or the points "move" along
  their scale and absorb the whole camera step: the Schur complement comes out exactly zero.
  The pin is not the answer there; the unknowns are. Cameras move by twists, points by
  directions, ideal points: `moved = motors << Direction`, a full-rank point block, no gauge,
  an exact solve. The gauge dyad stays where a point itself is the unknown.

## 11. Information and covariance

- The curvature of a cost over twists is an information form, `Scalar <- (Twist, Twist)`.
- Covariance is the map from a readout to the correlated twist, `Twist <- Line`, and moves
  like any map: `step << sigma(step >> Line)`.
- Readouts through the pairing; sampling by diagonalizing the readout form; the Schur
  complement as marginalization; the gain as a composition of maps.

## 12. Batches versus slots

- A batch axis indexes independent copies; a slot is an argument. A frame summed against its
  reciprocal is the coordinate spelling of a trace. Mostly a pointer to the syntax sheet.
