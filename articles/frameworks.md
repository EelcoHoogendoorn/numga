# Extensors, linear algebra, tensors and forms

This document compares geometric algebra and extensors with linear algebra, tensors, exterior algebra and differential forms. It sets out which objects and operations those frameworks share, which appear as special cases, and where additional structure enters. Together, the inclusions show how familiar constructions sit inside one algebra of elements, products and multilinear maps.

## The containment relationships

Extensors are multilinear maps between multivector spaces. Linear maps are the one-argument case, and every finite-dimensional linear map can be built from the products of the algebra. Ordinary tensors are a special case restricted to the generating vectors and their duals. Allowing all grades gives those maps areas, volumes, rotations and spinors as arguments and outputs.

The products themselves build these maps. For a fixed vector `a`, the expression `a ^ Vector` maps vectors to bivectors, while `Vector.dot(Vector)` and `Vector * Vector` are bilinear maps to scalars and even multivectors ([Extensors through multiple lenses](multiple_lenses.md)). Because the generating vector space sits inside its geometric algebra as the one-vectors, each of the other frameworks appears by restricting which subspaces and products enter:

| Framework or object | Its place in geometric algebra and extensors |
| --- | --- |
| Linear algebra | The one-argument extensors, including every linear map between the chosen spaces |
| Tensors on the generating vector space | Extensors restricted to vector and covector arguments |
| Tensor algebra | Sums, tensor products and contractions of those restricted maps |
| Exterior algebra | The multivectors with their wedge-product structure |
| Exterior forms | Scalar-valued extensors on the corresponding exterior powers |
| Multivectors | Extensors with no arguments |

These inclusions keep the constructions of the smaller frameworks intact while extending their arguments and outputs to every grade through the products of the algebra.

## Matrices and linear algebra

Every linear map between finite-dimensional multivector spaces is a linear extensor, and every such map can be built from geometric-algebra products, whether it acts from vectors to vectors, between different grades, or on mixed-grade spaces. A scalar readout multiplied by a fixed output gives a rank-one map. In projective space, a plane's reading of a point scales a fixed direction:

```python
dyad = direction * (plane & Point)                                # [] Antivector- <- Antivector-
```

Sums of these dyads build every linear map, because the complementary pairing gives a complete set of independent readouts and the output assigned to each readout is arbitrary. Products of independent readouts extend the same construction to every multilinear map, so the sum-of-products construction and the multilinear-map definition describe the same class of extensors ([Extensors through multiple lenses](multiple_lenses.md#3-the-mathematical-lens)).

Abstract linear algebra applies to multivector spaces as it does to any vector space, and here every linear map is built inside the algebra itself. Conventional matrix work concentrates on one-vector maps and inner-product pairings, introducing cross products, oriented volumes and spinor actions as separate operations. Geometric algebra and extensors connect them through one set of products. In geometric-algebra treatments of linear algebra [[1, §2]](#references), [[2, §1.2.3]](#references), extending a linear map on vectors to preserve the wedge product gives its outermorphism, whose action on the top grade defines the determinant. General linear extensors also include maps that change grade or act on mixed subspaces. Composition, inversion, trace, determinant and eigenspaces remain available, where `A.solve(b)` returns an element of the input space of `A` and `A.solve(B)` returns a map into that space with the inputs of `B`.

### Pairings and adjoints

In matrix calculations, a transpose often reverses the direction of a map while standing for two distinct operations. An adjoint transfers a map across a nondegenerate inner product, whereas a dual map pulls measurements of the output back to measurements of the input. For an orientation-preserving map `A` on intrinsic elements and an extrinsic measurement `c`, their defining equations separate the two pairings:

```python
A.adjoint()(b).scalar_product(a) == b.scalar_product(A(a))        # [] Scalar+
A.adjugate()(c) & a == c & A(a)                                   # [] Scalar+
```

The first uses the metric scalar product. The second uses the [complementary pairing](complementarity.md#claims) of [the regressive product](regressive_product.md), running from complements of the outputs to complements of the inputs and keeping the extrinsic factor first on both sides (so for a duality map `D` from intrinsic to extrinsic elements, `D.adjugate()(b) & a == D(a) & b` in every dimension). In standard linear-algebra texts [[3, §§3F, 7A]](#references), a matrix transpose in dual bases represents this dual map, and in real orthonormal bases it also represents the one-vector adjoint. Keeping the pairing explicit records which operation is meant and which spaces the resulting map acts between.

Because the complementary pairing uses no metric, the adjugate remains defined when the metric is degenerate or when the map itself has no inverse. A camera pulls an image condition back into the scene without an inverse camera, and extending a deformation to an exterior grade gives the cofactor map for [area transport](common_practice.md#surface-normals-under-deformation).

### Eigenvalue problems

An ordinary eigenvalue problem compares a map's output directly with its input, which requires both to lie in the same space. A map from a space to its complement, such as a stiffness or an inertia from motion rates to loads, instead needs a second form to define an eigenvalue comparison. Pairing both stiffness and inertia with a second motion rate exposes their symmetric energy forms and gives a symmetric generalized eigenproblem for [vibration modes](common_practice.md#vibration-modes) directly. Singular values likewise depend on norms chosen on the input and output spaces, such as an inertia or uncertainty form, rather than a coordinate dot product.

## Tensors and tensor algebra

A tensor on a vector space acts multilinearly on vectors and covectors, and geometric calculus extends tensors to multilinear functions on Clifford algebras [[4]](#references). Carrying that multilinear-map view across oriented and mixed-grade subspaces places ordinary tensors inside extensors as a special case. In an inner-product space, `a.dot(Vector)` gives a covector readout by pairing one-vectors; through the complementary pairing, a plane reads a point as `plane & point` and `plane & Point` is the linear form, which stays available when the metric is degenerate. Multiplying scalar-valued maps on independent arguments builds tensor products, multiplying by a fixed output gives a map with that output type, and binding arguments, composing maps and contracting paired arguments keep their standard meanings.

Because an extensor's arguments can belong to any subspace of the algebra, a multilinear map can also act directly on mixed-grade elements such as rotors (`Scalar+ + Bivector+`). Sandwiching rest-shape edge directions with the rotor identity map `Rotor` places `Rotor` on both sides of the product and forms a symmetric bilinear map on pairs of rotors:

```python
edge_forms = edges.dot(Rotor >> rest_edges)                       # [edges] Scalar+ <- (Rotor+, Rotor+)
```

In geometry-processing literature, as-rigid-as-possible surface modeling fits a local rotation at each vertex by minimizing weighted edge residuals over its incident edges [[5]](#references). In a [surface deformation](../examples/surfaces/arap/arap.ipynb) example, summing `edge_forms` at each vertex gives that best-fitting proper rotation directly as the form's top eigenvector. A [point-cloud registration](../examples/geometry/registration/registration.ipynb) example aligns rotations and translations together through a linear residual on projective motors, and [Discrete differential geometric algebra](ddga.md#up-the-grade-axis) differentiates fields of rotors on a simplicial topology in the same way.

### Grades and symmetries

An antisymmetric pair of tensor arguments can be represented by a single bivector argument whose grade carries the antisymmetry. [Inertia, vorticity and curvature](common_practice.md#inertia-vorticity-and-curvature) then act directly on planes of rotation and area patches, with an inertia mapping a motion rate to a momentum and Riemann curvature mapping an area patch to a plane of rotation. In spacetime-algebra treatments of general relativity [[6]](#references), torsion-free curvature is a symmetric linear bivector map, while pair symmetry and the algebraic Bianchi identity remain conditions on the map when torsion is present.

Symmetric tensors stay symmetric multi-argument maps rather than forming a new grade. Tensor notation is concise for stating permutation symmetries across many indices, whereas extensor expressions make symmetry direct when it comes from a geometric construction such as summed spring energies. Moving a map composes the action of a transformation on each input and output space, and for an extensor built from [geometric ingredients](multiple_lenses.md#2-the-geometric-lens) that transformation law follows from moving the ingredients.

### Reference and physical metrics

Tensor algebra usually starts from a bare vector space and supplies a metric as a separate bilinear form, whereas a geometric algebra builds a metric into its geometric product. That built-in metric need not be the physical metric of a model, because a position-dependent physical metric can be supplied as an extensor while the algebra's product stays fixed. With a fixed nondegenerate reference metric defining the dot product, any physical metric on a direction space `Direction` can be written as `Direction.dot(metric_map)` for a self-adjoint `metric_map`, or kept as a duality map from directions to planes evaluated through the regressive product, which rests on a unit of volume alone. Norms and rotors built from the algebra's own product belong to the reference metric.

Kerr black-hole examples put both choices side by side. A [metric formulation](../examples/relativity/kerr/kerr_metric.ipynb) example represents the curved metric as a duality map `Vector+ <- Antivector-` together with its Levi-Civita connection. Following the gauge-theory formulation of gravity in spacetime algebra [[7, §§1, 5.3]](#references), a [gauge formulation](../examples/relativity/kerr/kerr.ipynb) example instead represents the geometry by a position-dependent frame map `Antivector- <- Antivector-` and a bivector connection over flat Minkowski space. Both evaluate the same squared proper interval along a direction:

```python
metric_interval = metric(geometry)(direction) & direction         # [] Scalar+
gauge_interval = -gauge_inverse(geometry)(direction).squared()    # [] Scalar+
```

On the exterior Kerr domain traced in a [side-by-side comparison](../examples/relativity/kerr/kerr_comparison.ipynb) example, both formulations produce identical rays and polarization images. Placing the local metric in an extensor or in a frame over the algebra's product is therefore a choice of representation.

## Forms and exterior algebra

Exterior algebra sits inside geometric algebra as its wedge-product structure, with every grade and alternating product intact, while the geometric product adds metric contractions in the same space. An alternating form on several vectors corresponds here to a linear scalar-valued extensor on their exterior power, so a two-form reads a bivector and a higher form reads the corresponding higher-grade element through the regressive product.

In differential geometry [[8, §1]](#references), differential forms are fields of alternating tensors whose exterior derivative, pullback and integration require no metric, whereas Hodge duality enters as a separate map once a metric and volume orientation are given. In discrete electromagnetism on cell complexes [[9, §4]](#references), ordinary and twisted chains distinguish inner from outer orientations for quantities such as circulation, flux and charge; [Why orientation](orientation.md) carries the corresponding intrinsic and extrinsic distinction into geometric algebra.

### Exterior and metric structure in applications

Nearly all `numga` examples combine exterior and metric structure in one calculation. A swimming body balances forques and twists against a distance-dependent fluid drag, elastic waves combine oriented facet tractions with a metric shear modulus, and gravitational waves couple spacetime curvature on area patches with metric light-cone propagation. Electromagnetism shows the split clearly. In premetric electrodynamics [[10, §2]](#references), charge conservation and the homogeneous flux laws require no metric, while the [constitutive relation](common_practice.md#electromagnetic-constitutive-tensors) introduces the spacetime metric in vacuum or a medium's response in matter. Three-dimensional vector calculus packages those pieces into composite operators such as curl and the cross product, whereas geometric algebra keeps the exterior products and metric contractions distinct inside one algebra.

## Geometric algebra and extensors

A multivector is an extensor with no arguments, and naming an argument space by its type turns the products that combine multivectors into linear and [multilinear maps](multiple_lenses.md#2-the-geometric-lens). Geometric algebra supplies the elements and products; extensors carry those products into the maps between elements. In a [scenegraph](../examples/geometry/scenegraph/scenegraph.ipynb) example, a thin lens built from a centre, a plane and a focal length refracts projective rays, moves under a rigid motor by conjugating its output and input, and composes with robot-arm kinematics and sensor projection into one map per body. An inertia, a covariance or a material map has the same status and can be added, composed, moved or solved through the same operations.

## Geometric constructions

Linear and tensor formulations introduce oriented products, reciprocity pairings and spinor actions through separate auxiliary definitions. Levi-Civita contractions supply alternating structure beside tensor products, and robotics screw theory defines reciprocal wrenches and twists by a six-coordinate pairing of zero instantaneous power [[11, Ch. 2, §5.3]](#references). Each row below pairs one of those auxiliary definitions with the geometric-algebra product or extensor operation that gives it:

| Construction introduced separately | Construction in geometric algebra and extensors |
| --- | --- |
| Levi-Civita contractions for oriented products | Wedge and regressive products on the relevant grades |
| Cross product with a fixed vector in three dimensions | `a ^ Vector`, followed by the normal identification |
| Determinant | The outermorphism's action on volume elements |
| Cofactor construction for area transport in three dimensions | The outermorphism on oriented areas |
| Reciprocity pairing of twists and wrenches in screw theory | The regressive pairing of a twist with a forque |
| Pauli and Dirac matrices | Multiplication acting as a map on spinors |

The wedge that builds a cross product also builds a plane, the outermorphism that transports an area also transports a volume, and the product that acts on spinors in an [extensor representations](../examples/math/extensor_representations/extensor_representations.ipynb) example also composes rotations across dimensions.

## Transport and irreversible thermodynamics

In linear irreversible thermodynamics, flux densities relate to thermodynamic forces through a matrix of response coefficients whose entries obey Onsager–Casimir reciprocity when time-reversal-breaking fields such as a magnetic field are reversed [[12, §I]](#references). Thermoelectric and thermomagnetic treatments name cross-couplings such as the Peltier, Seebeck, Hall, Nernst, Ettingshausen and Righi–Leduc effects separately alongside magnetoresistance and Kelvin's relation.

[Geometry and quantities](thermodynamics.md) models coupled transport in one map by joining spacetime's geometric algebra with a null basis vector for each tracked extensive variable. When an entropy direction is joined to spacetime, each carrier combines its charge and carried entropy in one parcel. An advection map joins a carrier's velocity with that parcel into a flux, and its adjugate pulls the unified drive `-gradient ^ potentials` back into the force on the carrier, while a spatial mobility turns force into drift (with a magnetic plane turning that drift through the carrier's Hall angle), assembling into one transport response `Antibivector- <- Bivector+` from drives to fluxes.

Each of those named effects comes from scalar readings of that single map. Because the force on a carrier and the flux it makes are one advection map read backwards and forwards, symmetric spatial mobilities carry directly into Onsager's and Kelvin's relations, reversing the magnetic field turns the response into its adjugate as in Onsager–Casimir reciprocity, and pairing a drive with its flux gives a dissipation `drive & flux` that is not negative when the carriers' mobilities are positive, as a [magnetotransport](../examples/thermodynamics/magnetotransport/magnetotransport.ipynb) example computes on a planar bar and a Corbino ring.

In the matrix notation of linear irreversible thermodynamics, coupled electric and thermal transport in a plane with magnetic field $B$ has the form $J_i = \sum_j L_{ij}(B)\,X_j$ with $L_{ij}(B) = L_{ji}(-B)$ for equal-parity forces. In the planar carrier model above, scalar readings of the single response give the sixteen entries for two currents in two directions.

## Specialized algebras and calculi

Many named algebras in physics, geometry and engineering are taught as separate formalisms with their own symbols, rules and libraries. Within geometric algebra and extensors they can be represented by choosing a signature, a grade subspace and an orientation ([Complementarity](complementarity.md#appendix-representations)), first at the multivector level and then as maps when extensor arguments enter; `numga` examples linked in the tables below run those constructions.

### Nullary subalgebras and grade subspaces

Each row below gives the signature, grade subspace and orientation at the multivector level that represent one of these number systems, Lie algebras or geometries from the literature:

| Algebra or system | Representation in multivectors |
| --- | --- |
| Complex, split-complex and dual numbers ($\mathbb{C}$, $\mathbb{R}[\varepsilon]/(\varepsilon^2=0)$) [[13]](#references) | Even subalgebra (`Scalar+ + Bivector+`) of `x+ y+`, `x+ t-` and `x+ w0` |
| Quaternions [[14]](#references) and biquaternions ($\mathbb{H}$, $\mathbb{H}\otimes\mathbb{C}$) | Even subalgebras of `x+ y+ z+` and `x+ y+ z+ t-` |
| Dual quaternions and screw / motor algebra ($\mathrm{SE}(3)$, $\mathfrak{se}(3)$, $\mathfrak{se}(3)^*$) [[15]](#references) | Motors (`Rotor+`), twists (`Bivector+`) and forques (`Antibivector-`) in `w0 x+ y+ z+` |
| Plücker coordinates and Grassmann–Cayley algebra [[15]](#references) | Meets of planes (`Bivector+`) and joins of points (`Antibivector-`) in projective space |
| Lie algebras $\mathfrak{so}(p,q)$, $\mathfrak{spin}(p,q)$ and duals $\mathfrak{g}^*$ [[16]](#references) | Bivectors (`Bivector+`) under `.commutator()` and `.exp()`, paired with momenta (`Antibivector-`) |
| Cayley–Klein geometries (elliptic, hyperbolic, Euclidean, Galilean, Minkowski) [[17]](#references) | Projective algebras with chosen metric signature (`w+`, `w-`, `w0` beside spatial and temporal axes) |
| Conformal geometry and Möbius transformations [[18]](#references), [[2, §1.1]](#references) | Null-cone points, spheres, circles, planes and $\mathrm{Spin}(4,1)$ versors in `x+ y+ z+ e+ f-` |
| Lie sphere geometry in three dimensions [[19]](#references) | Oriented spheres on the null quadric in $\mathrm{Cl}(4,2)$ and totally null contact two-planes |
| Pauli and spacetime spinor/bivector algebras [[20]](#references) | `x+ y+ z+` and even subalgebra of `x+ y+ z+ t-`, with intrinsic spins (`Antibivector-`) and extrinsic fields (`Bivector+`) |
| Twistors and conformal spacetime spinors ($\mathrm{SU}(2,2) \cong \mathrm{Spin}_0(4,2)$) [[21]](#references) | Spinors of conformal spacetime $\mathrm{Cl}(4,2)$, with real null twistors tracing light rays ([twistors](../examples/relativity/twistors/twistors.ipynb)) |

### Extensor constructions

Each row below supplies multivector subspaces as inputs and outputs to represent one of these operator or tensor calculi:

| Calculus or formalism | Representation in extensors and `numga` examples |
| --- | --- |
| Pauli and Dirac matrix representations ($\sigma_i$, $\gamma_\mu$, Weyl / Majorana spinors) [[20]](#references) | Left multiplication on minimal left ideals into `Spinor+ <- Spinor+` maps ([extensor representations](../examples/math/extensor_representations/extensor_representations.ipynb), [spinors](../examples/relativity/spinors/spinors.ipynb)) |
| Octonions, split octonions, $\mathfrak{g}_2$ derivations and $\mathrm{Spin}(8)$ triality ($\mathbb{O}$, $\mathbb{O}_s$) [[22]](#references) | Bilinear multiplication on `Scalar+ + Vector+` in $\mathrm{Cl}(0,7)$ via an invariant trivector, with $\mathfrak{g}_2$ as its stabilizer ([octonions](../examples/math/octonions/octonions.ipynb), [octonionic rolling](../examples/math/g2_rolling/g2_rolling.ipynb), [triality](../examples/math/triality/triality.ipynb)) |
| Bloch-ball states, POVM effects and Pauli transfer channels [[23]](#references) | Unit-weight projective points (`Antivector-`), effect planes (`Vector+`) and channels `Antivector- <- Antivector-` ([process tomography](../examples/quantum/process_tomography/process_tomography.ipynb)) |
| Jones and Stokes polarization calculus [[24]](#references) | Screen slices (`Vector+`), screen rotors and polarization coherency `Antivector- <- Vector+` ([Kerr polarization](../examples/relativity/kerr/kerr.ipynb)) |
| Multiparticle quantum registers and stabilizer codes [[25]](#references) | Direct-sum signatures ($\mathrm{Cl}(3n,0)$ or $\mathrm{Cl}(3n,n)$), commuting blade projectors and multilinear state maps ([two spins](../examples/quantum/two_spins/two_spins.ipynb), [stabilizer code](../examples/quantum/stabilizer/stabilizer.ipynb)) |
| Projective quadrics, conics and absolute metrics [[26]](#references) | Polarities `Vector+ <- Antivector-` and `Antivector- <- Vector+`, pulled back through projections by `.adjugate()` ([multiview reconstruction](../examples/estimation/multiview/multiview_reconstruction.ipynb)) |
| Algebraic relationships of discrete exterior calculus [[27]](#references) | Coboundaries, simplex gradients and material pullbacks by `.adjugate()` ([surface connections](../examples/surfaces/connections/connections.ipynb), [Discrete differential geometric algebra](ddga.md)) |
| Symplectic / Poisson brackets and Onsager–GENERIC thermodynamics [[28]](#references) | Duality maps `Vector+ <- Antivector-` and `Antivector- <- Vector+` on null state spaces ([GENERIC](../examples/thermodynamics/generic/generic.ipynb), [Ruppeiner](../examples/thermodynamics/ruppeiner/ruppeiner.ipynb)), and transport responses `Antibivector- <- Bivector+` ([magnetotransport](../examples/thermodynamics/magnetotransport/magnetotransport.ipynb)) |
| Premetric electrodynamics [[10]](#references) and crystal optics | Constitutive maps `Antibivector- <- Bivector+` from fields to excitations, with intrinsic scalar chiral couplings ([Fresnel wave surfaces](../examples/electromagnetism/fresnel/fresnel.ipynb)) |
| Voigt fourth-order elasticity and compliance [[29]](#references) | Multilinear stiffness and compliance maps relating strain, traction and crystal waves ([elastic stress](../examples/mechanics/stress/stress.ipynb), [crystal waves](../examples/mechanics/crystal_waves/crystal_waves.ipynb)) |

## Numerical methods

Standard numerical linear-algebra solvers, including Gauss–Newton, Schur complements, conjugate gradients, least squares and generalized eigensolvers, act directly on typed linear and bilinear extensors and return geometric updates such as a camera twist or a body's vibration mode. Keeping the pairings and argument spaces explicit preserves the geometric types of Schur blocks, pairs conjugate-gradient steps with cost differentials rather than coordinate vectors, and exposes symmetric quadratic costs directly before an artificial symmetrization or squared residual changes the problem ([Extensors and common practice](common_practice.md)).

## Conclusion

Geometric algebra and extensors place multivectors, linear maps, tensors, exterior products and differential forms inside one typed algebra. Operations that require separate definitions and sign rules when introduced one by one, such as alternating contractions, cross products, area cofactors, screw reciprocity and spinor actions, come from the same geometric products and extend to maps of any grade and arity. A model can therefore move from its elements to its constitutive laws, transformations and solvers within one algebraic language.

## Companion documents

* [Extensors through multiple lenses](multiple_lenses.md): how multilinear maps arise from partial products, transform under motions and assemble from dyads.
* [Why orientation](orientation.md): why geometric constructions and positive physical energies require intrinsic and extrinsic orientation.
* [The regressive product](regressive_product.md): how the regressive product pairs orientations, joins points, contracts fields and builds linear maps and their adjugates.
* [Complementarity](complementarity.md): how complementary pairings and physical representations extend across geometry, mechanics, estimation and physics.
* [Extensors and common practice](common_practice.md): practical comparisons between conventional coordinate formulations and typed extensor constructions.
* [Discrete differential geometric algebra](ddga.md): calculus and physics on simplicial topologies built from the simplex gradient.
* [Geometry and quantities](thermodynamics.md): equilibrium and transport thermodynamics in algebras extended with null directions for extensive quantities.
* `numga` documentation: [extensor syntax](../docs/extensor_syntax.md), [conventions](../docs/conventions.md), [moving maps](../docs/tutorial/extensors/08-moving-maps.md) and [the adjugate](../docs/tutorial/extensors/10-the-adjugate.md).

## References

1. [*The Design of Linear Algebra and Geometry*](https://davidhestenes.net/geocalc/pdf/DLAandG.pdf) (David Hestenes, *Acta Applicandae Mathematicae*, 1991).
2. [*Geometric Algebra for Computer Science: An Object-Oriented Approach to Geometry*](https://geometricalgebra.org/downloads/ga4cs_chapter1.pdf) (Leo Dorst, Daniel Fontijne and Stephen Mann, Morgan Kaufmann, 2007; open Chapter 1).
3. [*Linear Algebra Done Right*](https://linear.axler.net/LADR4e.pdf) (Sheldon Axler, 4th ed., 2024).
4. [*A Unified Language for Mathematics and Physics*](https://davidhestenes.net/geocalc/pdf/UnifiedLang.pdf) (David Hestenes, in *Clifford Algebras and Their Applications in Mathematical Physics*, 1986).
5. [*As-Rigid-As-Possible Surface Modeling*](https://igl.ethz.ch/projects/ARAP/arap_web.pdf) (Olga Sorkine and Marc Alexa, *Symposium on Geometry Processing*, 2007).
6. [*Curvature Calculations with Spacetime Algebra*](https://davidhestenes.net/geocalc/pdf/Curv_cal.pdf) (David Hestenes, *International Journal of Theoretical Physics*, 1986).
7. [*Gravity, gauge theories and geometric algebra*](https://arxiv.org/pdf/gr-qc/0405033) (Anthony Lasenby, Chris Doran and Stephen Gull, *Philosophical Transactions of the Royal Society A*, 1998; arXiv, 2004).
8. [*Lecture Notes on General Relativity*](https://arxiv.org/pdf/gr-qc/9712019) (Sean M. Carroll, 1997).
9. [*Discretization of Electromagnetic Problems: The "Generalized Finite Differences" Approach*](https://people.math.ethz.ch/~hiptmair/Courses/CEM/BOS05c.pdf) (Alain Bossavit, *Handbook of Numerical Analysis*, 2005).
10. [*Recent developments in premetric classical electrodynamics*](https://arxiv.org/pdf/physics/0610221) (Friedrich W. Hehl, Yakov Itin and Yuri N. Obukhov, 2006).
11. [*A Mathematical Introduction to Robotic Manipulation*](https://www.cds.caltech.edu/~murray/books/MLS/pdf/mls94-complete.pdf) (Richard M. Murray, Zexiang Li and S. Shankar Sastry, 1994).
12. [*Onsager Relations in Coupled Electric, Thermoelectric and Spin Transport: The Ten-Fold Way*](https://arxiv.org/pdf/1207.1629) (Philippe Jacquod, Robert S. Whitney, Jonathan Meair and Markus Büttiker, *Physical Review B*, 2012).
13. [*Erlangen Programme at Large 3.1: Hypercomplex Representations of the Heisenberg Group and Mechanics*](https://arxiv.org/pdf/1005.5057) (Vladimir V. Kisil, 2010).
14. [*On Quaternions*](https://www.maths.tcd.ie/pub/HistMath/People/Hamilton/Quatern2/Quatern2.pdf) (William Rowan Hamilton, *Philosophical Magazine*, 1844–1850).
15. [*On the Homogeneous Model of Euclidean Geometry*](https://arxiv.org/pdf/1101.4542) (Charles G. Gunn, 2011).
16. [*Lie groups as spin groups*](https://davidhestenes.net/geocalc/pdf/LGasSG.pdf) (Chris Doran, David Hestenes, Frank Sommen and Nadine Van Acker, *Journal of Mathematical Physics*, 1993).
17. [*Trigonometry of spacetimes: a new self-dual approach to a curvature/signature (in)dependent trigonometry*](https://arxiv.org/pdf/math-ph/9910041) (Francisco J. Herranz, Ramón Ortega and Mariano Santander, *Journal of Physics A*, 2000).
18. [*Generalized Homogeneous Coordinates for Computational Geometry*](https://davidhestenes.net/geocalc/pdf/CompGeom-ch2.pdf) (Hongbo Li, David Hestenes and Alyn Rockwood, in *Geometric Computing with Clifford Algebras*, 2001).
19. [*Taut and Dupin Submanifolds*](https://library.slmath.org/books/Book32/files/cecil.pdf) (Thomas E. Cecil, in *Tight and Taut Submanifolds*, MSRI Publications, 1997).
20. [*Spacetime Physics with Geometric Algebra*](https://davidhestenes.net/geocalc/pdf/SpacetimePhysics.pdf) (David Hestenes, *American Journal of Physics*, 2003).
21. [*Lectures on twistor theory*](https://arxiv.org/pdf/1712.02196) (Tim Adamo, *PoS Modave2017*, 2018).
22. [*The Octonions*](https://arxiv.org/pdf/math/0105155) (John C. Baez, *Bulletin of the American Mathematical Society*, 2002).
23. [*Introduction to Quantum Gate Set Tomography*](https://arxiv.org/pdf/1509.02921) (Erik Nielsen, John King Gamble, Kenneth Rudinger, Travis Scholten, Kevin Young and Robin Blume-Kohout, *Quantum*, 2021).
24. [*Polarimetric characterization of Light and media*](https://arxiv.org/pdf/2002.04105) (José J. Gil, *European Physical Journal Plus*, 2020).
25. [*Stabilizer Codes and Quantum Error Correction*](https://arxiv.org/pdf/quant-ph/9705052) (Daniel Gottesman, Caltech Ph.D. thesis, 1997).
26. [*Self-Calibration Supported Robust Projective Structure-from-Motion*](https://arxiv.org/pdf/2007.02045) (Viktor Larsson et al., 2020).
27. [*Discrete Exterior Calculus*](https://arxiv.org/pdf/math/0508341) (Mathieu Desbrun, Anil N. Hirani, Melvin Leok and Jerrold E. Marsden, 2005).
28. [*GENERIC: Review of Successful Applications and a Challenge for the Future*](https://arxiv.org/pdf/1810.08470) (Hans Christian Öttinger, 2018).
29. [*The constitutive tensor of linear elasticity: its decompositions, Cauchy relations, null Lagrangians, and wave propagation*](https://arxiv.org/pdf/1208.1041) (Giacomo Caviglia and Angelo Morro, 2012).
