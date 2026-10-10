# Common practice and the typed form

Everything below can be done with matrices, vectors and indices, and the arithmetic can be the same. What differs is which form is convenient. Each case pairs a documented common practice with the construction that is convenient here, and, where there is one, with the example that uses it.

## Maps, forms and pairings

These cases concern maps: how they pair, compose, move and decompose. With typed slots and no transpose, the form a problem wants is the one that is easiest to write, and the habitual forms are awkward or cannot be written at all.

**Normal equations and transposes in bundle adjustment.** The [multiview example](../examples/estimation/multiview/core.py) was first written with a coefficient transpose, as Gauss-Newton is taught: Jacobians, the normal matrix $J^T J$, and the right-hand side $J^T r$.

```python
h = (j.transpose()(j)).sum(axis=0)                   # Twist <- Twist
rhs = -(j.transpose()(res)).sum(axis=0)              # Twist
step = h.lstsq(rhs, rcond=1e-4)
```

The transpose identified a map's output with its input by index label, a Euclidean metric on coefficients hidden in the interface, and the residual needed a metric of its own. Removing the transpose from numga (commit `d0b41af`) left no way to write this, and the form that remained is the cone value itself: the curvature is a cone joined with the motion, the gradient the cone at the point joined with the motion, with no residual and no residual metric.

```python
curvature = (local_cones(motion) & motion).sum(axis=0)        # Scalar <- (Twist, Twist)
gradient = (local_cones(local_points) & motion).sum(axis=0)   # Scalar <- Twist
step = curvature.lstsq(-gradient, rcond=1e-4)                 # Twist
```

The Schur variant changed the same way. Its transposed version formed the point blocks as maps and inverted them with a pseudoinverse, which needed a gauge on homogeneous points; the typed version moves points by directions and solves the point curvature against the cross term, with the twist slot carried, `h_pt.solve(h_cross)`.

**Orthogonality of twists and wrenches.** Hybrid force and motion control was built on orthogonal complements of twist and wrench spaces, with a dot product on six coordinates. J. Duffy, [The fallacy of modern hybrid control theory that is based on "orthogonal complements" of twist and wrench spaces](https://onlinelibrary.wiley.com/doi/10.1002/rob.4620070202) (Journal of Robotic Systems, 1990), showed that this dot product mixes units and depends on the frame, and that the meaningful pairing of a twist is with a wrench. In numga twists and forques are different types, and the pairing between them is the regressive product, which needs no metric. The conjugate gradients of the [odometry example](../examples/estimation/odometry/core.py) take every inner product they need between a twist and a line, and precondition with a map of type `Line <- Twist` per pose; a dot product of two twists, the one that mixes radians with meters, is not available to write.

**Normals under a transform.** The standard practice in graphics is the normal matrix, the inverse transpose of the model matrix. It needs an inverse, fails for a singular transform, and flips normals under reflections. N. Reed, [Normals and the Inverse Transpose, Part 1: Grassmann Algebra](https://www.reedbeta.com/blog/normals-inverse-transpose-part-1/), and [Transforming normals: adjugate transpose vs inverse transpose](https://www.forwardscattering.org/post/62) argue for the adjugate, from the observation that a normal is a bivector. Here that is the default: a map on vectors moves bivectors by its outermorphism, and a map on points moves planes by its adjugate, defined for singular maps such as a camera.

```python
on_planes = (Plane & Point).solve(Plane & collineation)   # Plane <- Plane, singular collineation included
```

**Area elements and stress measures.** Continuum mechanics moves area elements through a deformation $F$ by Nanson's formula, $n\, da = J F^{-T} N\, dA$, and keeps a zoo of stress measures, Cauchy, first and second Piola-Kirchhoff, and Kirchhoff, converted into one another by factors of $F$, $F^{-T}$ and $J$; keeping track of which factor goes where is a well-known source of mistakes. An area element is a bivector, and a deformation moves it by its outermorphism, the two edges moved together. The [area transport example](../examples/mechanics/area_transport/core.py) carries oriented patches this way, and gets the cofactor that Nanson's formula needs as the adjoint of the adjugate of that map, without an inverse, at zero volume included:

```python
areas = deformation.outermorphism(Bivector)          # Bivector <- Bivector
cofactor = areas.adjugate().adjoint()                 # Vector <- Vector: area normals carried forward
```

The stress measures follow the same way. The Cauchy stress is a map from oriented areas of the deformed body to forces. Composed with the area map, it takes areas of the undeformed body to forces, the first Piola-Kirchhoff stress, and composed further with the inverse deformation on its output, it takes them to forces in the undeformed frame, the second. The factors of $J$ and $F^{-T}$ are what the composition with the area map is in coordinates.

**Modes from a map that is not symmetric.** A stiffness assembled as a map from twists to forques, rates to momentum lines, is not symmetric in coordinates, since its input and output are different kinds of things. The habitual remedies are symmetrizing, which discards information, or the normal equations, which square the condition number and cost a product. The [modes example](../examples/mechanics/modes/core.py) pairs the output with an open twist instead, which permutes the map into the symmetric form it is, and solves the generalized eigenproblem on the two forms directly.

```python
pe_form = Twist & stiffness                     # Scalar <- (Twist, Twist)
ke_form = Twist & inertia                       # Scalar <- (Twist, Twist)
values, modes = pe_form.eigh(ke_form)
```

**Fitting a rotation.** The accepted method is Kabsch's: the SVD of a cross-covariance matrix, with a documented correction, flipping a sign by the determinant, because the SVD can return a reflection. The correction is easy to get wrong in practice; see for instance the [reflection fix in MapClosures](https://github.com/PRBonn/MapClosures/pull/118). Horn's and Davenport's methods avoid it with a quaternion eigenproblem, on a 4×4 matrix assembled by hand. In the [registration example](../examples/geometry/registration/core.py) the fit is a form on rotors, the rotor open on both sides of the sandwich, and its top eigenvector is a rotor by type, so no reflection can come out of it.

```python
alignment = target.scalar_product(Rotor >> source).sum(axis=0)   # Scalar <- (Rotor, Rotor)
values, rotors = alignment.eigh()
```

**Maps in a moving frame.** The constitutive relations of a moving medium, Minkowski's, are among the more involved formulas of electrodynamics: the medium's response at rest, rewritten for an observer it moves past, mixes the electric and magnetic parts of both the field and the excitation. A constitutive relation is a map from fields to excitations, and moving a map is a composition: the field pulled into the rest frame, the response at rest, the excitation pushed back out. The [constitutive example](../examples/electromagnetism/constitutive/core.py) writes it in one line, and the [moving charge example](../examples/electromagnetism/moving_charge/core.py) builds the field gradients of a charge at any speed the same way, from the charge at rest:

```python
rotor >> base_medium(rotor << Bivector)                       # Antibivector <- Bivector
charge.boost >> (radial + turning)(charge.boost << Vector)   # Bivector <- Vector
```

The same composition moves an inertia between body and space frames, the place where $R I R^T$ and $R^T I R$ get confused; local inputs, global outputs, or both, are three different compositions with three different types, as in the [extensor tutorial's chapter on moving maps](tutorial/extensors/08-moving-maps.md).

## Tensors and their encodings

These cases concern tensors with many slots or with antisymmetric parts, and the encodings that tensor practice uses to fit them into vectors and matrices.

**Antisymmetric objects encoded as vectors.** In three dimensions, index notation turns an antisymmetric object into a vector through the Levi-Civita symbol, and the encoding leaves traces in the formulas. The inertia tensor is the standard case: $I_{ij} = \sum m\,(r^2 \delta_{ij} - r_i r_j)$, with a trace subtracted, takes an angular velocity vector to an angular momentum vector. Both are planes of rotation encoded as their normals, and the $r^2 \delta_{ij}$ term is what the encoding costs; it has no counterpart in other dimensions. As a map on the planes themselves, the inertia is each mass point joined with its own motion under an open rate, with nothing subtracted, as in the [spinning top example](../examples/mechanics/spinning_top/core.py):

```python
(points & points.commutator(Line) * masses).sum()     # momentum <- rate
```

Fluid mechanics has the same encoding with a factor in it: the vorticity vector, the curl of the velocity, is twice the angular velocity of a fluid element, and the spin tensor, the antisymmetric part of the velocity gradient, is half the dual of the vorticity, a standard place to lose a factor of two. In the [vortices example](../examples/mechanics/vortices/core.py) the derivative of the flow is one even element, the divergence plus the vorticity bivector. The Riemann tensor has pair symmetries, antisymmetric in each pair of indices and symmetric between the pairs, which index notation states as identities; as a symmetric map on bivectors, the antisymmetry is the type and only the symmetry remains to state. The Petrov classification works with it in that form, and the [curvature example](../examples/relativity/curvature/core.py) types it as `Bivector <- Bivector` and binds an observer into it for the tidal map.

**Fourth-order tensors flattened into matrices.** To compute with the elasticity tensor, engineering practice flattens it into a 6×6 matrix in Voigt notation, with the shear components of strain doubled, the engineering shear strain. The flattened matrix does not keep the tensor's norm or eigenvalues, which Mandel notation repairs with factors of $\sqrt{2}$; P. Helnwein, [Some remarks on the compressed matrix representation of symmetric second-order and fourth-order tensors](https://www.sciencedirect.com/science/article/abs/pii/S0045782500002632) (Computer Methods in Applied Mechanics and Engineering, 2001), sorts out the covariant and contravariant readings these representations need. The [crystal waves example](../examples/mechanics/crystal_waves/core.py) builds the stiffness of a cubic crystal with three open vector slots and computes with it in that form, binding a heading into two slots to get the map whose eigenvalues are the wave speeds; nothing is flattened, so no factor is to be placed.

```python
crystal(heading, Vector, heading).eigh()                # density times squared speed, polarization
```

**Which index is which.** Textbooks differ on whether the first index of the stress tensor is the direction of the force or the normal of the face, as the [notes on traction and stress](https://archive.nptel.ac.in/content/storage2/courses/105106049/lecnotes/mainch4.html) of NPTEL point out. For the symmetric Cauchy stress the choice does not change a result; for the first Piola-Kirchhoff stress, which is not symmetric, it decides the formula. With typed slots the question does not arise: a stress is a map from oriented areas to forces, and its input and output have different types.

**The constitutive tensor without a metric.** Tensor electrodynamics has a premetric formulation, in F. W. Hehl and Y. N. Obukhov, [Foundations of Classical Electrodynamics: Charge, Flux, and Metric](https://books.google.com/books/about/Foundations_of_Classical_Electrodynamics.html?id=48-hHXL-CYUC) (2003). Maxwell's equations themselves need no metric; the constitutive tensor, with 36 components, takes field strengths to excitations, both 2-forms, and all the metric there is enters through it. The metric is not avoided, it is located: in a vacuum the constitutive map is the metric's duality, and in a medium it replaces it. That is the form an extensor takes without being asked: the [constitutive example](../examples/electromagnetism/constitutive/core.py) types the medium as `Antibivector <- Bivector`, a map from field bivectors to their complements, six by six, and writes the field equations of a plane wave as wedges, `(k ^ Bivector) + (k ^ medium).dual()`, with the metric only in the medium.

## The algebra itself

These cases are benefits of geometric algebra itself rather than of extensors: they concern the objects, not maps between them, and hold with or without open slots.

**Spatial vectors.** R. Featherstone's spatial vector algebra, in A Beginner's Guide to 6-D Vectors (IEEE Robotics & Automation Magazine, 2010) and his [notes on spatial vector algebra](http://royfeatherstone.org/teaching/2008/notes.pdf), keeps motion and force vectors apart with two cross products, `crm` and `crf`, and transforms forces by $X^* = X^{-T}$; screw theory pairs them through a swap of the two 3-vector halves. Here a twist and a forque are a bivector and a line of the same algebra: one commutator acts on both, one sandwich moves both, and the swap is the regressive product, a signed permutation of coefficients.

**Blending rigid motions.** Linear blend skinning, the standard in real-time animation, blends transformation matrices, and a blend of rigid transforms is not rigid: the candy-wrapper collapse at twisting joints is its well-known artifact. L. Kavan, S. Collins, J. Žára and C. O'Sullivan, [Skinning with dual quaternions](https://dl.acm.org/doi/abs/10.1145/1230100.1230107) (I3D 2007), blend in the even subalgebra instead, and the artifact goes. Dual quaternions are the motors of projective geometric algebra; in that algebra the blend of motors is the natural one to write, and the matrix blend is not.

**Fields under a boost.** The transformation of the electric and magnetic fields between frames, in J. D. Jackson's Classical Electrodynamics, splits each field into parts parallel and perpendicular to the velocity, with factors of $\gamma$ and cross products between the two fields, and is easy to garble. The field is one bivector, and a boost moves it by one sandwich.

**Composition of boosts.** Two boosts in different directions compose to a boost and a rotation, and the rotation is what Thomas precession accumulates. Getting it wrong is part of the history of the electron's spin: the spin-orbit coupling of Uhlenbeck and Goudsmit came out twice the observed value until L. H. Thomas, [The motion of the spinning electron](https://www.nature.com/articles/117514a0) (Nature, 1926), supplied the relativistic factor of one half, and the derivation remains a standard place to stumble. With boosts as rotors, the composition is their product, and the rotation is its part in the planes of space.

**Pseudovectors under reflection.** The magnetic field, angular momentum and torque are pseudovectors: under a reflection they pick up an extra sign that ordinary vectors do not, a rule to remember for every parity argument. As bivectors they transform by the same sandwich as every other element, and the sign comes out of the product.

**Quaternion conventions.** Attitude estimation in robotics and aerospace uses two quaternion multiplications, Hamilton's and a flipped one associated with JPL, together with active and passive readings and world-to-body or body-to-world frames. H. Sommer, I. Gilitschenski, M. Bloesch, S. Weiss, R. Siegwart and J. Nieto, [Why and How to Avoid the Flipped Quaternion Multiplication](https://www.mdpi.com/2226-4310/5/3/72) (Aerospace, 2018), document the confusion and how to migrate between conventions. A rotor has one product, and moving a value forward and back is `>>` and `<<`.
