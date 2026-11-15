# Extensors, linear algebra, tensors and forms

This document compares geometric algebra and extensors with linear algebra, tensors, exterior algebra and differential forms. It sets out how those frameworks relate: which objects and operations they share, which appear as special cases, and where additional structure enters. Together, the inclusions show how familiar constructions sit inside one algebra of elements, products and multilinear maps.

## The containment relationships

Extensors are multilinear maps between multivector spaces. Linear maps are the one-argument case, and every finite-dimensional linear map can be constructed from the products of the algebra. Ordinary tensors are a special case, restricted to the generating vectors and their duals; allowing all grades gives those maps areas, volumes, rotations and spinors as arguments and outputs.

The products themselves construct these maps. For a fixed vector `a`, the expression `a ^ Vector` maps vectors to bivectors, while `Vector.dot(Vector)` and `Vector * Vector` are bilinear maps to scalars and even multivectors, as [Extensors through multiple lenses](multiple_lenses.md) develops. Because the generating vector space sits inside its geometric algebra as the one-vectors, the other frameworks appear by restricting which subspaces and products enter:

| Framework or object | Its place in geometric algebra and extensors |
| --- | --- |
| Linear algebra | The one-argument extensors, including every linear map between the chosen spaces |
| Tensors on the generating vector space | Extensors restricted to vector and covector arguments |
| Tensor algebra | Sums, tensor products and contractions of those restricted maps |
| Exterior algebra | The multivectors with their wedge-product structure |
| Exterior forms | Scalar-valued extensors on the corresponding exterior powers |
| Multivectors | Extensors with no arguments |

These inclusions preserve the constructions of the smaller frameworks while extending their arguments and outputs to every grade through the products of the algebra.

## Matrices and linear algebra

Every linear map between finite-dimensional multivector spaces is a linear extensor, and every such map can be built from geometric-algebra products, whether it acts from vectors to vectors, between different grades, or on mixed-grade spaces. A scalar readout multiplied by a fixed output gives a rank-one map; in projective space, a plane's reading of a point scales a fixed direction:

```python
dyad = direction * (plane & Point)                                # [] Antivector- <- Antivector-
```

Sums of these dyads construct every linear map, because the complementary pairing supplies a complete set of independent readouts and the output assigned to each readout is arbitrary. Products of independent readouts extend the same construction to every multilinear map, and the [completeness argument](multiple_lenses.md#3-the-mathematical-lens) shows that the sum-of-products construction and the multilinear-map definition describe the same class of extensors.

Abstract linear algebra applies to multivector spaces as it does to any vector space, and here every linear map is constructive within the algebra itself. Conventional matrix work concentrates on one-vector maps and inner-product pairings, introducing cross products, oriented volumes and spinor actions as separate operations; geometric algebra and extensors connect them through one set of products. Extending a linear map on vectors to preserve the wedge product gives its outermorphism, whose action on the top grade defines the determinant ([The Design of Linear Algebra and Geometry](https://davidhestenes.net/geocalc/pdf/DLAandG.pdf), §2), whereas general linear extensors also include maps that change grade or act on mixed subspaces. Composition, inversion, trace, determinant and eigenspaces remain available: `A.solve(b)` returns an element of the input space of `A`, and `A.solve(B)` returns a map into that space with the inputs of `B`, ready to combine with further geometric products.

### Pairings and adjoints

In matrix calculations, a transpose often serves as a routine way to reverse the direction of a map, standing for two distinct constructions: an adjoint transfers a map across an inner product, whereas a dual map pulls measurements of the output back to measurements of the input. Their defining relationships separate the two pairings:

```python
A.adjoint()(b).scalar_product(a) == b.scalar_product(A(a))        # [] Scalar+
A.adjugate()(c) & a == c & A(a)                                   # [] Scalar+
```

The first uses the metric scalar product. The second uses the metric-free complementary pairing of [Complementarity](complementarity.md#claims), running from complements of the outputs to complements of the inputs. In dual bases a matrix transpose represents this dual map, and in real orthonormal bases it also represents the one-vector adjoint ([Linear Algebra Done Right](https://linear.axler.net/LADR4e.pdf), §§3F and 7A); keeping the pairing explicit records which operation is meant and which spaces the resulting map acts between.

Because the complementary pairing uses no metric, the adjugate remains defined when the metric is degenerate or when the map itself is singular: a camera pulls an image condition back into the scene without an inverse camera, and extending a deformation to an exterior grade gives the cofactor map for area transport, as the [surface-normals case](common_practice.md#surface-normals-under-deformation) develops.

### Eigenvalue problems

An ordinary eigenvalue problem compares a map's output directly with its input, which requires both to inhabit the same space, whereas a map from a space to its complement, such as a stiffness or inertia from motion rates to loads, needs a second form to define an eigenvalue comparison. Pairing both stiffness and inertia with a second rate exposes their symmetric energy forms and produces a symmetric generalized eigenproblem directly, examined in [Vibration modes](common_practice.md#vibration-modes). Singular values likewise depend on norms chosen on the input and output spaces, such as an inertia or uncertainty form, and keeping those forms in the construction avoids replacing them silently with a coordinate dot product.

## Tensors and tensor algebra

A tensor on a vector space acts multilinearly on vectors and covectors, and geometric calculus generalizes tensors to multilinear functions on Clifford algebras ([A Unified Language for Mathematics and Physics](https://davidhestenes.net/geocalc/pdf/UnifiedLang.pdf)). Carrying that multilinear-map view across oriented and mixed-grade subspaces places ordinary tensors inside extensors as a special case. In an inner-product space, `a.dot(Vector)` supplies a covector readout by pairing one-vectors; through the complementary pairing, a plane reads a point as `plane & point` and `plane & Point` is the linear form, which remains available when the metric is degenerate. Multiplying scalar-valued maps on independent arguments builds tensor products, multiplying by a fixed output gives a map with that output type, and binding arguments, composing maps and contracting paired arguments keep their standard meanings.

Because an extensor's arguments can occupy any subspace of the algebra, a multilinear map can also act directly on mixed-grade elements such as rotors (`Scalar+ + Bivector+`). Sandwiching rest-shape edge directions with the rotor identity map `Rotor` places `Rotor` on both sides of the product, turning the quadratic action into a symmetric bilinear form on pairs of rotors:

```python
edge_forms = edges.dot(Rotor >> rest_edges)                       # [edges] Scalar+ <- (Rotor+, Rotor+)
```

The local step of [As-Rigid-As-Possible Surface Modeling](https://igl.ethz.ch/projects/ARAP/arap_web.pdf) fits a rotation at each vertex by minimizing weighted edge residuals over its incident edges; in the [surface-deformation example](../examples/surfaces/arap/arap.ipynb), summing `edge_forms` at each vertex gives that best-fitting proper rotation as the form's top eigenvector. The [point-cloud registration example](../examples/geometry/registration/registration.ipynb) aligns rotations and translations together through a linear residual on projective motors, and [Discrete differential geometric algebra](ddga.md#up-the-grade-axis) extends the same differentiation to fields of rotors on a simplicial topology.

### Grades and symmetries

An antisymmetric pair of tensor arguments can be represented by a single bivector argument whose grade carries the antisymmetry, so an inertia acts directly on a plane of rotation and Riemann curvature maps an area patch to a plane of rotation, examined in [Inertia, vorticity and curvature](common_practice.md#inertia-vorticity-and-curvature); without torsion it is a symmetric linear bivector map ([Curvature Calculations with Spacetime Algebra](https://davidhestenes.net/geocalc/pdf/Curv_cal.pdf)), while pair symmetry and the algebraic Bianchi identity remain conditions on the map when torsion is present. Symmetric tensors stay symmetric multi-argument maps rather than forming a new grade: tensor notation is concise for stating permutation symmetries across many indices, while extensor expressions make symmetry direct when it comes from a geometric construction such as summed spring energies. Moving a map composes the action of a transformation on each input and output space, and for an extensor built from geometric ingredients that transformation law follows automatically from moving the ingredients, as [Extensors through multiple lenses](multiple_lenses.md#2-the-geometric-lens) explains.

### Reference and physical metrics

Tensor algebra usually starts with a bare vector space and supplies a metric as a separate bilinear form, whereas a geometric algebra builds a metric into its geometric product. That built-in metric need not be the physical metric of a model: a position-dependent physical metric can be supplied as an extensor while the algebra's product stays fixed. With a fixed nondegenerate reference metric defining the dot product, any physical metric on a direction space `Direction` can be written as `Direction.dot(metric_map)` for a self-adjoint `metric_map`, or kept as a duality map from directions to covectors evaluated through `&`, which uses no metric and rests on a unit of volume alone; norms and rotors built from the algebra's own product belong to the reference metric.

The Kerr black-hole notebooks put both choices side by side. The [metric formulation](../examples/relativity/kerr/kerr_metric.ipynb) carries the curved metric as a duality map `Vector+ <- Antivector-` together with its Levi-Civita connection, while the [gauge formulation](../examples/relativity/kerr/kerr.ipynb) carries a position-dependent frame map `Antivector- <- Antivector-` and a bivector connection over a flat reference spacetime ([Gravity, gauge theories and geometric algebra](https://arxiv.org/pdf/gr-qc/0405033), §§1 and 5.3), where the Kerr solution and its curvature bivector map are formulated as fields on Minkowski space. Both evaluate the same squared proper interval along a direction:

```python
metric_interval = metric(geometry)(direction) & direction         # [] Scalar+
gauge_interval = -gauge_inverse(geometry)(direction).squared()    # [] Scalar+
```

On the exterior Kerr domain traced in the [side-by-side comparison](../examples/relativity/kerr/kerr_comparison.ipynb), both formulations produce identical rays and polarization images, showing that for local geometry placing the metric in an extensor or in a frame over the algebra's product is a choice of representation.

## Forms and exterior algebra

Exterior algebra sits inside geometric algebra as its wedge-product structure, with every grade and alternating product intact, while the geometric product adjoins metric contractions inside the same space. An alternating form on several vectors corresponds here to a linear scalar-valued extensor on their exterior power: a two-form measures a bivector and a higher form measures the corresponding higher-grade element, evaluated without a metric through `&`.

Differential forms are fields of alternating tensors whose exterior derivative, pullback and integration require no metric, whereas Hodge duality enters as a separate map once a metric and volume orientation are supplied ([Lecture Notes on General Relativity](https://arxiv.org/pdf/gr-qc/9712019), §1). On oriented cell complexes, ordinary and twisted chains distinguish inner from external orientations for quantities such as circulation, flux and charge ([Discretization of Electromagnetic Problems: The "Generalized Finite Differences" Approach](https://people.math.ethz.ch/~hiptmair/Courses/CEM/BOS05c.pdf), §4); [Why orientation](orientation.md) carries a corresponding intrinsic and extrinsic distinction into geometric algebra.

### Exterior and metric structure in applications

Nearly all of the examples in this repository combine exterior and metric structure in the same calculation: a swimming body balances forques and twists against a distance-dependent fluid drag, elastic waves combine oriented facet tractions with a metric shear modulus, and gravitational waves couple spacetime curvature on area patches with metric light-cone propagation. Electromagnetism shows the division cleanly: charge conservation and the homogeneous flux laws are metric-free ([Recent developments in premetric classical electrodynamics](https://arxiv.org/pdf/physics/0610221), §2), while the constitutive relation introduces the spacetime metric in vacuum or a medium's response in matter to fix wave propagation, as [Electromagnetic constitutive tensors](common_practice.md#electromagnetic-constitutive-tensors) examines as an extensor. Three-dimensional vector calculus packages those pieces into composite operators such as curl and the cross product, whereas geometric algebra keeps the exterior products and metric contractions distinct within one algebra so the field, derivative, constitutive map and observer combine directly.

## Geometric algebra and extensors

A multivector is an extensor with no arguments, and naming an argument space by its type turns the same geometric products that combine multivectors into linear and multilinear maps, as [Extensors through multiple lenses](multiple_lenses.md#2-the-geometric-lens) develops. Geometric algebra supplies the elements and products; extensors carry those products into the maps between elements. In the [scenegraph example](../examples/geometry/scenegraph/scenegraph.ipynb), a thin lens built from a centre, a plane and a focal length refracts projective rays, moves under a rigid motor by conjugating its output and input, and composes with robot-arm kinematics and sensor projection into one map per body; an inertia, covariance or material tensor has the same status and can be added, composed, moved or solved through the same operations.

## Geometric constructions

Linear and tensor formulations introduce oriented products, reciprocity pairings and spinor actions through separate auxiliary definitions. Levi-Civita contractions supply alternating structure beside tensor products, and screw theory defines reciprocal wrenches and twists by a six-coordinate pairing of zero instantaneous power ([A Mathematical Introduction to Robotic Manipulation](https://www.cds.caltech.edu/~murray/books/MLS/pdf/mls94-complete.pdf), Ch. 2 §5.3). In geometric algebra and extensors each of those operations comes from the algebra's own products:

| Construction introduced separately | Construction in geometric algebra and extensors |
| --- | --- |
| Levi-Civita contractions for oriented products | Wedge and regressive products on the relevant grades |
| Cross product with a fixed vector in three dimensions | `a ^ Vector`, followed by the normal identification |
| Determinant | The outermorphism's action on volume elements |
| Cofactor construction for area transport in three dimensions | The outermorphism on oriented areas |
| Reciprocity pairing of twists and wrenches in screw theory | The regressive pairing of a twist with a forque |
| Pauli and Dirac matrices | Multiplication acting as a map on spinors |

The wedge that builds a cross product also builds a plane, the outermorphism that transports an area also transports a volume, and the product that acts on spinors in [extensor representations](../examples/math/extensor_representations/extensor_representations.ipynb) also composes rotations, carrying consistent sign conventions across dimensions.

## Transport and irreversible thermodynamics

Linear irreversible thermodynamics relates flux densities to thermodynamic forces through a matrix of linear response coefficients whose entries obey Onsager–Casimir reciprocity when time-reversal-breaking fields such as a magnetic field are reversed ([Onsager Relations in Coupled Electric, Thermoelectric and Spin Transport: The Ten-Fold Way](https://arxiv.org/pdf/1207.1629), §I), while thermoelectric and thermomagnetic treatments name cross-couplings such as the Peltier, Seebeck, Hall, Nernst, Ettingshausen and Righi–Leduc effects separately alongside magnetoresistance and Kelvin's relation.

[Geometry and quantities](thermodynamics.md) develops coupled transport in one map by joining spacetime's geometric algebra with a degenerate basis vector for each tracked extensive variable. When an entropy direction is joined to spacetime, each carrier species combines its charge and carried entropy in one vector, one exterior derivative `-gradient ^ potentials` pushes each carrier according to its contents while a magnetic plane turns its drift through its Hall angle, and sweeping those contents along the drift sums to a single transport response `Antibivector- <- Bivector+` from drives to fluxes.

Each of those effects comes from scalar readings of that single map. For symmetric carrier mobilities, Kelvin's relation holds because one content vector couples each carrier to both drive and flux, reversing the magnetic field turns the response into its adjugate as in Onsager–Casimir reciprocity, and pairing a drive with its flux gives a dissipation `drive & flux` that is not negative when the mobilities are positive, as the [magnetotransport example](../examples/thermodynamics/magnetotransport/magnetotransport.ipynb) computes across a planar bar and a Corbino ring.

In the matrix notation of linear irreversible thermodynamics, coupled electric and thermal transport in a plane with magnetic field $B$ has the form $J_i = \sum_j L_{ij}(B)\,X_j$ with $L_{ij}(B) = L_{ji}(-B)$ for equal-parity forces; in the planar carrier model above, scalar readings of the single response give the sixteen entries for two currents in two directions.

## Specialized algebras and calculi

Many named algebras in physics, geometry and engineering are taught as separate formalisms with their own symbols, rules and libraries. Within geometric algebra and extensors they can be represented by choosing a signature, a grade subspace and an orientation, as the [representations appendix of Complementarity](complementarity.md#appendix-representations) catalogues, first at the nullary level of multivectors and then as maps when extensor arguments enter.

### Nullary subalgebras and grade subspaces

Choosing a signature, a grade subspace and an orientation at the multivector level represents the specialized number systems, Lie algebras and geometries below:

| Algebra or system | Representation in multivectors |
| --- | --- |
| [Complex, split-complex and dual numbers](https://arxiv.org/pdf/1005.5057) ($\mathbb{C}$, $\mathbb{R}[\varepsilon]/(\varepsilon^2=0)$) | The even subalgebra (`Scalar+ + Bivector+`) of 2D Euclidean (`x+ y+`), Lorentzian (`x+ t-`) and degenerate (`x+ w0`) spaces |
| [Quaternions](https://www.maths.tcd.ie/pub/HistMath/People/Hamilton/Quatern2/Quatern2.pdf) and biquaternions ($\mathbb{H}$, $\mathbb{H}\otimes\mathbb{C}$) | The even subalgebras of 3D space (`x+ y+ z+`, $\mathrm{Cl}^+(3,0)$) and spacetime (`x+ y+ z+ t-`), respectively; the latter includes scalars, bivectors and the pseudoscalar |
| [Dual quaternions and screw / motor algebra](https://arxiv.org/pdf/1101.4542) ($\mathrm{SE}(3)$, $\mathfrak{se}(3)$, $\mathfrak{se}(3)^*$) | Even motors (`Rotor+`), extrinsic twists (`Bivector+`) and intrinsic wrenches or forques (`Antibivector-`) in 3D projective space (`w0 x+ y+ z+`), paired by `twist & forque` |
| [Plücker coordinates and Grassmann–Cayley algebra](https://arxiv.org/pdf/1101.4542) | Meets of planes (`^`, `Bivector+`) and joins of points (`&`, `Antibivector-`) in projective space |
| [Lie algebras](https://davidhestenes.net/geocalc/pdf/LGasSG.pdf) $\mathfrak{so}(p,q)$, $\mathfrak{spin}(p,q)$ and Lie–Poisson duals $\mathfrak{g}^*$ | Extrinsic bivectors (`Bivector+`) under `.commutator()` and `.exp()`, paired by `&` with intrinsic momenta (`Antibivector-`) in $\mathfrak{g}^*$ |
| [Cayley–Klein geometries](https://arxiv.org/pdf/math-ph/9910041) (elliptic, hyperbolic, Euclidean, Galilean, Minkowski) | Projective geometry with a chosen signature and absolute structure; with positive spatial directions, `w+`, `w-` and `w0` give elliptic, hyperbolic and Euclidean geometry |
| [Conformal geometry and Möbius transformations](https://davidhestenes.net/geocalc/pdf/CompGeom-ch2.pdf) | Null-cone points, spheres, circles, planes and $\mathrm{Spin}(4,1)$ versors in 3D conformal space (`x+ y+ z+ e+ f-`, $\mathrm{Cl}(4,1)$) |
| [Lie sphere geometry](https://library.slmath.org/books/Book32/files/cecil.pdf) in three dimensions | Oriented spheres on the projective null quadric in $\mathrm{Cl}(4,2)$, with oriented contact represented by totally null two-planes |
| [Pauli and spacetime spinor/bivector algebras](https://davidhestenes.net/geocalc/pdf/SpacetimePhysics.pdf) | 3D space (`x+ y+ z+`) and the even subalgebra of spacetime (`x+ y+ z+ t-`), with spins modelled as intrinsic patches (`Antibivector-`) and fields as extrinsic planes (`Bivector+`) |
| [Twistors and conformal spacetime spinors](https://arxiv.org/pdf/1712.02196) ($\mathrm{SU}(2,2) \cong \mathrm{Spin}_0(4,2)$) | Spinor representations of conformal spacetime $\mathrm{Cl}(4,2)$, with real null twistors determining light rays ([twistors and linked light](../examples/relativity/twistors/twistors.ipynb)) |

### Extensor constructions

Supplying multivector subspaces as inputs and outputs represents the operator and tensor calculi below:

| Calculus or formalism | Representation in extensors |
| --- | --- |
| [Pauli and Dirac matrix representations](https://davidhestenes.net/geocalc/pdf/SpacetimePhysics.pdf) ($\sigma_i$, $\gamma_\mu$, Majorana / Weyl spinors) | Left multiplication on suitable left ideals (`state_readout(Full * Spinor * projector)`) gives faithful `Spinor+ <- Spinor+` maps and trace readouts; Weyl subspaces carry the even action ([extensor representations](../examples/math/extensor_representations/extensor_representations.ipynb), [spinors](../examples/relativity/spinors/spinors.ipynb)) |
| [Octonions](https://arxiv.org/pdf/math/0105155), split octonions, $\mathfrak{g}_2$ derivations and $\mathrm{Spin}(8)$ triality ($\mathbb{O}$, $\mathbb{O}_s$) | Bilinear multiplication maps `((Octonion * Octonion) * (1 - orientation)).cast(Octonion)` in $\mathrm{Cl}(0,7)$ or Cayley–Dickson doubling across even and odd grades of $\mathrm{Cl}(3,0)$, with $\mathfrak{g}_2$ as the kernel of `Bivector.commutator(orientation)` ([octonions](../examples/math/octonions/octonions.ipynb), [octonionic rolling](../examples/math/g2_rolling/g2_rolling.ipynb), [triality](../examples/math/triality/triality.ipynb)) |
| [Bloch-ball states, POVM effects and Pauli transfer channels](https://arxiv.org/pdf/1509.02921) | Projective points of unit weight (`Antivector-`), measurement planes (`Vector+`) paired by `effect & state`, and linear maps `Antivector- <- Antivector-` ([process tomography](../examples/quantum/process_tomography/process_tomography.ipynb)) |
| [Jones and Stokes polarization calculus](https://arxiv.org/pdf/2002.04105) | Transverse screen slices (`Vector+`), rotor and projection maps on screen directions, and the polarization coherency `Antivector- <- Vector+` on the screen subspace ([Kerr polarization](../examples/relativity/kerr/kerr.ipynb)) |
| Multiparticle quantum registers and [stabilizer codes](https://arxiv.org/pdf/quant-ph/9705052) | Direct-sum signatures ($\mathrm{Cl}(3n,0)$ or $\mathrm{Cl}(3n,n)$), commuting blade involutions `(1 + blade) / 2` as stabilizer projectors, and multilinear maps across state spaces ([two spins](../examples/quantum/two_spins/two_spins.ipynb), [stabilizer code](../examples/quantum/stabilizer/stabilizer.ipynb)) |
| [Projective quadrics, conics and absolute metrics](https://arxiv.org/pdf/2007.02045) | Polarities `Vector+ <- Antivector-` and `Antivector- <- Vector+`, pulled back through singular projections by `.adjugate()` ([multiview reconstruction](../examples/estimation/multiview/multiview_reconstruction.ipynb)) |
| Algebraic relationships of [discrete exterior calculus](https://arxiv.org/pdf/math/0508341) | Coboundary and material pullback maps on fields, complementary pairings and adjugates, with chain identities and Laplacians built by composition ([surface connections](../examples/surfaces/connections/connections.ipynb), [Discrete differential geometric algebra](ddga.md)) |
| Symplectic / Poisson brackets and [Onsager–GENERIC thermodynamics](https://arxiv.org/pdf/1810.08470) | Maps `Vector+ <- Antivector-` (symplectic 2-form or Ruppeiner metric) and `Antivector- <- Vector+` (Poisson bivector or friction mobility) on a null-metric state space ([GENERIC](../examples/thermodynamics/generic/generic.ipynb), [Ruppeiner](../examples/thermodynamics/ruppeiner/ruppeiner.ipynb)), and transport responses `Antibivector- <- Bivector+` ([magnetotransport](../examples/thermodynamics/magnetotransport/magnetotransport.ipynb)) |
| [Premetric electrodynamics](https://arxiv.org/pdf/physics/0610221) and crystal optics | Constitutive maps `Antibivector- <- Bivector+` representing linear relations from field 2-forms to twisted excitation 2-forms, including `Antipseudoscalar-` chiral couplings ([Fresnel wave surfaces](../examples/electromagnetism/fresnel/fresnel.ipynb)) |
| [Voigt fourth-order elasticity](https://arxiv.org/pdf/1208.1041) and compliance | Fourth-order stiffness and compliance extensors relating strain (`Vector+ <- Antivector-`), Cauchy stress (`Antivector- <- Vector+`) and crystal wave propagation ([elastic stress](../examples/mechanics/stress/stress.ipynb), [crystal waves](../examples/mechanics/crystal_waves/crystal_waves.ipynb)) |

## Numerical methods

Standard numerical linear-algebra solvers, including Gauss–Newton, Schur complements, conjugate gradients, least squares and generalized eigensolvers, operate directly on typed linear and bilinear extensors from construction through solution, returning geometric updates such as a camera twist or a body's vibration mode. Keeping the pairings and argument spaces explicit throughout an algorithm preserves the geometric types of Schur blocks, pairs conjugate-gradient steps with cost differentials rather than coordinate vectors, and exposes symmetric quadratic costs directly before an artificial symmetrization or squared residual alters the problem, as [Extensors and common practice](common_practice.md) develops case by case.

## Closure

Geometric algebra and extensors place multivectors, linear maps, tensors, exterior products and differential forms inside one typed algebra. Operations that require separate definitions and sign rules when introduced one by one, such as alternating contractions, cross products, area cofactors, screw reciprocity and spinor actions, arise from the same geometric products and extend to maps of any grade and arity, so a model moves from its elements to its constitutive laws, transformations and solvers without changing language.

Companion documents examine [extensors from three viewpoints](multiple_lenses.md), [orientation under reflections](orientation.md), [complementary pairings](complementarity.md), [physical representations](complementarity.md#appendix-representations), [discrete calculus on topologies](ddga.md) and [joined thermodynamic algebras](thermodynamics.md); library specifics are covered in [extensor syntax](../docs/extensor_syntax.md), [conventions](../docs/conventions.md), [moving maps](../docs/tutorial/extensors/08-moving-maps.md) and [the adjugate](../docs/tutorial/extensors/10-the-adjugate.md).
