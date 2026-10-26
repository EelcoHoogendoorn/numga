# Extensors, Linear Algebra, Tensors and Forms

This document compares geometric algebra and extensors with linear algebra, tensors, exterior algebra and differential forms. It is intended for readers familiar with these frameworks who want to understand how they relate: which objects and operations they share, which are special cases of broader constructions, and where additional structure is needed. The aim is to show how familiar mathematics fits into the combined algebra, and what follows from having its objects, products and maps available together.

Background on [extensors](multiple_lenses.md), [syntax](extensor_syntax.md), [conventions](conventions.md) and [orientation](why_orientation.md) is developed separately.

## The containment relationships

Extensors are multilinear maps between multivector spaces. Linear maps are the one-argument case, and every finite-dimensional linear map can be constructed from the products of the algebra. Ordinary tensors are a special case, restricted to the generating vectors and their duals. Allowing all grades gives those maps a richer geometric vocabulary: areas, volumes, rotations and spinors can be arguments and outputs in their own right.

The products themselves construct these maps. For a fixed vector `a`, the expression `a ^ Vector` maps vectors to bivectors; `Vector.dot(Vector)` and `Vector * Vector` are bilinear maps to scalars and even multivectors. The [multiple-lenses document](multiple_lenses.md) develops this construction. The relationships to the other frameworks follow from the spaces they admit and the products available on them.

The generating vector space sits inside its geometric algebra as the one-vectors. Extensors extend multilinear maps from that space to arbitrary multivector spaces, including mixtures of grades.

| Framework or object | Its place in GA and extensors |
| --- | --- |
| Linear algebra | The one-argument extensors, including every linear map between the chosen spaces |
| Tensors on the generating vector space | Extensors restricted to vector and covector arguments |
| Tensor algebra | Sums, tensor products and contractions of those restricted maps |
| Exterior algebra | The multivectors with their wedge-product structure |
| Exterior forms | Scalar-valued extensors on the corresponding exterior powers |
| Multivectors | Extensors with no arguments |

These inclusions preserve the constructions of the smaller frameworks. Extensors extend their arguments and outputs to every grade, and GA supplies products that relate those grades.

## Matrices and linear algebra

Every linear map between finite-dimensional multivector spaces is a linear extensor, and every such map can be built from GA products. This includes maps between different grades and maps on mixed-grade spaces, as well as the familiar maps from vectors to vectors.

A scalar readout multiplied by a fixed output is a rank-one map. A plane's reading of a point can scale a fixed direction:

```python
dyad = direction * (plane & Point)
```

Sums of these dyads construct every linear map. The complementary pairing supplies a complete set of independent readouts, and the output assigned to each readout is arbitrary. Products of independent readouts extend the construction to every multilinear map. The [completeness argument](multiple_lenses.md#3-the-mathematical-lens-multilinear-maps-and-sums-of-products) establishes that the sum-of-products construction and the multilinear-map definition describe the same class of extensors.

Abstract linear algebra applies to multivector spaces as it does to any vector space. The inclusion here is constructive: all its linear maps are available through the products of GA. Conventional matrix work concentrates on one-vector maps and inner-product pairings, with cross products, oriented volumes and spinor actions introduced as further constructions. GA and extensors contain that collection of operations and connect them through the same algebra.

Composition, inversion, trace, determinant and eigenspaces remain available. `A.solve(b)` gives an element of the input space of `A`; `A.solve(B)` gives a map into that space with the inputs of `B`. The result participates in the same geometric products as the maps that defined it.

### Pairings and adjoints

In matrix calculations, a transpose often appears as a routine way to reverse the direction of a map. Two constructions lie behind that notation. An adjoint transfers a map across an inner product; a dual map pulls measurements of the output back to measurements of the input. Their defining relationships make the pairings explicit:

```python
A.adjoint()(b).scalar_product(a) == b.scalar_product(A(a))
A.adjugate()(c) & a == c & A(a)
```

The first uses the scalar product. The second uses the complementary pairing and runs from complements of the outputs to complements of the inputs; this is the [adjugate](tutorial/extensors/10-the-adjugate.md) used in the examples. The transpose represents the dual map in dual bases, and in orthonormal Euclidean bases it also represents the one-vector adjoint. Keeping the pairing explicit records which of these operations is intended and which spaces the resulting map acts on. [Linear Algebra Done Right](https://linear.axler.net/LADR4e.pdf), §§3F and 7A, develops the dual-map and adjoint constructions.

The complementary pairing remains available with a degenerate metric. It also lets a singular map pull measurements back: a camera can carry an image condition back into the scene without an inverse camera. The cofactor construction follows from extending a map to the appropriate exterior grade. The [normals case](common_practice.md#surface-normals-under-deformation) develops the relation to area transport.

### Eigenvalue problems

An ordinary eigenvalue problem compares a map's output with its input, which requires both to belong to the same space. A bilinear form needs a second form to define an eigenvalue comparison. Pairing a rate-to-load map with another rate exposes the energy form and its symmetry. Stiffness and inertia then give a symmetric generalized eigenproblem directly, as the [vibration case](common_practice.md#vibration-modes) shows.

Singular values likewise depend on norms on the input and output spaces. An inertia or uncertainty form may supply the relevant measure. Keeping these forms in the construction makes the choices visible and avoids replacing them silently by a default norm.

## Tensors and tensor algebra

A tensor on a vector space acts multilinearly on vectors and covectors. Those arguments are among the arguments an extensor can take, so ordinary tensors are contained in extensors. An extensor can additionally take a bivector, an oriented volume or a spinor directly, and return an object of any multivector type.

A covector is represented by its scalar readout of an argument. In the inner-product formulation, `a.dot(Vector)` supplies that readout by pairing one-vectors. The complementary pairing supplies readouts without relying on the metric; a plane measures a point through `plane & point`, and `plane & Point` is the corresponding linear form. This also makes covector arguments available when the metric is degenerate.

The tensor product is carried by independent arguments. Multiplying scalar-valued maps, each with its own argument, gives a multilinear scalar-valued map; multiplying by a fixed output gives a multilinear map with that output type. Sums of these products span the tensors. Binding arguments, composing maps and contracting paired arguments retain their familiar meanings. Extensors make these operations available with the full set of GA products and multivector types.

### Grades and symmetries

An antisymmetric pair of tensor arguments can be represented by one bivector argument. Its grade carries the antisymmetry. An inertia can therefore act directly on a plane of rotation, and curvature can take an area to a plane of rotation. The [inertia and curvature cases](common_practice.md#inertia-vorticity-and-curvature) develop the practical difference. Curvature's pair symmetry and Bianchi identity remain conditions on that map.

Symmetric tensors remain symmetric maps: a symmetric product of two vectors does not become a new grade. Tensor notation is particularly concise for stating permutation symmetries among many arguments. Extensor expressions are particularly direct when the symmetry follows from a geometric construction, as it does for sums of quadratic spring energies. The grades and the symmetries of the map perform different jobs.

Tensor transformation laws also retain their meaning. Each argument and output has a space, and moving the map composes the corresponding action with those spaces. An extensor built from geometric ingredients transforms with its ingredients under motions preserving its products. The [moving-maps tutorial](tutorial/extensors/08-moving-maps.md) develops that relationship; the [multiple-lenses account](multiple_lenses.md#2-the-geometric-lens-meaning-from-construction) explains why the transformation law follows from the construction.

### Reference and physical metrics

Tensor algebra usually starts with a vector space and supplies a metric as a separate bilinear form. A geometric algebra includes a metric in its product. That built-in metric need not be the physical metric of the model: a varying physical metric can be supplied as an extensor while the algebra's product stays fixed.

With a fixed nondegenerate reference metric defining the dot product, every physical metric on a direction space `Direction` has the form `Direction.dot(metric_map)`. The position-dependent `metric_map` is self-adjoint with respect to the reference metric, and its variation can describe curved geometry. Alternatively, the physical metric can remain a map from directions to covectors, with evaluation through the metric-independent complementary pairing. GA supports either placement; choosing its product does not commit the model to that metric for its physical measurements.

The Kerr notebooks make this concrete. The [metric formulation](../examples/relativity/kerr/kerr_metric.ipynb) carries the physical metric and its Levi-Civita connection explicitly. The [gauge formulation](../examples/relativity/kerr/kerr.ipynb) carries a position-dependent frame map and a bivector connection, with a fixed flat metric on the local directions. The frame takes local directions to coordinate directions; its inverse brings a direction back to the local frame for measurement. The two formulations read the same squared interval:

```python
metric_interval = metric(geometry)(direction) & direction
gauge_interval = -gauge_inverse(geometry)(direction).squared()
```

The [side-by-side comparison](../examples/relativity/kerr/kerr_comparison.ipynb) follows the same rays and polarization through both descriptions. The choice changes which map carries the geometry and how transport is expressed. Supplying a metric separately or including a reference metric in the product is a choice of formulation; neither restricts the physical metrics or geometric constructions that can be described.

## Forms and exterior algebra

Exterior algebra is contained in GA as its wedge-product structure. All its grades and their alternating products are available. The geometric product combines this exterior structure with metric contractions, so a calculation can use both within the same algebra.

An exterior form is also a special case of an extensor. An alternating form on several vectors is a linear scalar-valued map on their exterior power: a two-form measures a bivector, and a higher form measures the corresponding higher-grade element. The complementary pairing represents these scalar evaluations without a metric. The Hodge star is a further map supplied by the metric and orientation.

Differential forms are fields of these alternating forms. Their exterior derivative, pullback and integration retain their roles in conservation laws. Twisted forms record the orientation behavior needed by quantities such as charge and flux. Intrinsic and extrinsic orientations carry the corresponding distinction through geometric expressions; [orientation](why_orientation.md) develops it, and [Bossavit's treatment of computational electromagnetism](https://www.jpier.org/ac_api/download.php?id=00080102) gives the connection to forms and oriented meshes.

### Exterior and metric structure in applications

Roughly 90–95% of the examples in this collection use both metric and exterior structure. A swimming body needs force and torque balance together with a fluid response that depends on distance. Elastic waves need oriented faces and flux together with a metric in the shear response. Gravitational curvature acts on spacetime areas and produces the tidal forces whose accelerations the metric determines. The need for both structures comes from the problems themselves.

Electromagnetism makes their relationship particularly clear. Exterior structure describes circulation, flux and conservation; the vacuum constitutive law supplies the metric structure that determines light propagation. The separation is explicit in [premetric electrodynamics](https://arxiv.org/abs/physics/0610221): the field equations keep their exterior form while the constitutive map supplies the response. The [constitutive-tensor case](common_practice.md#electromagnetic-constitutive-tensors) expresses that response as an extensor.

Three-dimensional vector calculus packages these structures into separate operators. Curl combines an exterior derivative with a metric and orientation identification; cross products encode oriented planes through their normal vectors. The exterior structure is present in these constructions even when the notation names only vectors. GA keeps the exterior products and metric contractions together, so their combination follows from the algebra's operations. The constitutive map, field, derivative and observer can participate in the same geometric calculation.

For a practitioner, the combined algebra should be the starting point. Introducing either part alone and supplying the other through separate operators leaves the connections to be reconstructed for each application. Learning the products together makes those connections available when the physical model needs them.

## Geometric algebra and extensors

A multivector is an extensor with no arguments. The same products that combine multivectors construct maps when some factors vary over a space. GA supplies the geometric objects and their products; extensors carry those products into the maps between objects.

A thin lens makes that relationship concrete. It is an object built from a centre, a plane and a focal length, and a map acting on rays. A motor moves it by acting on its input and output:

```python
lens = Line - (center & (Line ^ plane)) / focal
placed_lens = motor >> lens(motor << Line)
```

The [scenegraph example](../examples/geometry/scenegraph/core.py) moves lenses, applies them to rays and composes them into a camera. An inertia, covariance or material law has the same status: it can be added, composed, moved or solved as a map, with its geometric meaning supplied by its construction.

## Geometric constructions

Linear and tensor formulations commonly introduce oriented products, reciprocity pairings and spinor actions through separate definitions. Levi–Civita contractions are a familiar example: they supply alternating structure to an expression whose other operations are tensor products and contractions. In GA that structure is available through the exterior products themselves.

| Construction introduced separately | Construction in GA and extensors |
| --- | --- |
| Levi–Civita contractions for oriented products | Wedge and regressive products on the relevant grades |
| Cross product with a fixed vector in three dimensions | `a ^ Vector`, followed by the normal identification |
| Determinant | The outermorphism's action on volume elements |
| Cofactor construction for area transport in three dimensions | The outermorphism on oriented areas |
| Reciprocity pairing of twists and wrenches in screw theory | The regressive pairing of a twist with a wrench |
| Pauli and Dirac matrices | Multiplication acting as a map on spinors |

Each row connects a familiar construction to an operation used throughout the algebra. The wedge that constructs a cross product also constructs a plane; the outermorphism that carries an area also carries a volume; multiplication on spinors comes from the product that also composes rotations. The [representations example](../examples/math/extensor_representations/core.py) constructs the spinor actions directly.

The common origin matters. These constructions acquire their geometric meaning and conventions together, and the grades carry the same relationships into other dimensions.

## Specialized algebras and calculi

Many named algebras in physics, geometry and engineering are taught as separate formalisms with their own symbols, rules and libraries. Within GA and extensors they arise by choosing a signature, a grade subspace and an orientation—first at the nullary level of multivectors, and then as maps and correlations when extensor arguments enter.

### Nullary subalgebras and grade subspaces

| Algebra or system | Its place in multivectors |
| --- | --- |
| Complex, split-complex and dual numbers ($\mathbb{C}$, $\mathbb{R}[\varepsilon]/(\varepsilon^2=0)$) | The even subalgebra (`Scalar + Bivector`) of 2D Euclidean (`x+y+`), Lorentzian (`x+t-`) and degenerate (`x+w0`) algebras |
| Quaternions and biquaternions ($\mathbb{H}$, $\mathbb{H}\otimes\mathbb{C}$) | The even subalgebras of 3D space (`x+y+z+`, $\mathrm{Cl}^+(3,0)$) and spacetime (`STA`), respectively; the latter includes scalars, bivectors and the pseudoscalar |
| Dual quaternions and screw / motor algebra ($\mathrm{SE}(3)$, $\mathfrak{se}(3)$, $\mathfrak{se}(3)^*$) | Even motors (`Scalar + Bivector + Pseudoscalar`), extrinsic twists (`Bivector+`) and intrinsic wrenches/forques (`Antibivector-`) in `PGA3D`, paired by `twist & forque` |
| Plücker coordinates and Grassmann–Cayley algebra | Meets of planes (`^`, `Bivector+`) and joins of points (`&`, `Antibivector-`) in projective space (`PGA`) |
| Lie algebras $\mathfrak{so}(p,q)$, $\mathfrak{spin}(p,q)$ and Lie–Poisson duals $\mathfrak{g}^*$ | Extrinsic bivectors (`Bivector+`) under `.commutator()` and `.exp()`, paired by `&` with intrinsic momenta (`Antibivector-`) in $\mathfrak{g}^*$ |
| Cayley–Klein geometries (elliptic, hyperbolic, Euclidean, Galilean, Minkowski) | Projective geometry with a chosen full signature and absolute structure; with positive spatial directions, `w+`, `w-`, `w0` give elliptic, hyperbolic and Euclidean geometry |
| Conformal geometry and Möbius transformations | Null-cone points, spheres, circles, planes and $\mathrm{Spin}(4,1)$ versors in `CGA3D` ($\mathrm{Cl}(4,1)$) |
| Lie sphere geometry in three dimensions | Oriented spheres on the projective null cone of $\mathrm{Cl}(4,2)$, with contact represented by totally null two-planes ([Lie sphere geometry](https://link.springer.com/article/10.1007/s13366-018-0394-6)) |
| Pauli and spacetime spinor/bivector algebras | `VGA3D` (`x+y+z+`) and the even subalgebra of `STA` (`x+y+z+t-`), with spins as intrinsic patches (`Vector-` in space, `Bivector-` in spacetime) and fields as extrinsic planes (`Bivector+`) |
| Twistors and conformal spacetime spinors ($\mathrm{SU}(2,2) \cong \mathrm{Spin}_0(4,2)$) | Spinor representations of conformal spacetime $\mathrm{Cl}(4,2)$, with null twistors determining light rays ([twistors and linked light](../examples/relativity/twistors/core.py)) |

### Extensor constructions

| Calculus or formalism | Its place in extensors |
| --- | --- |
| Pauli and Dirac matrix representations ($\sigma_i$, $\gamma_\mu$, Majorana / Weyl spinors) | Left multiplication on suitable left ideals (`state_readout(Full * Spinor * projector)`) gives faithful `Spinor <- Spinor` maps and trace/Fierz readouts; Weyl subspaces carry the even action ([extensor representations](../examples/math/extensor_representations/scenarios.py), [spinors](../examples/relativity/spinors/core.py)) |
| Octonions, split octonions, $\mathfrak{g}_2$ derivations and $\mathrm{Spin}(8)$ triality ($\mathbb{O}$, $\mathbb{O}_s$) | Bilinear multiplication maps `((Octonion * Octonion) * (1 - orientation)).cast(Octonion)` in $\mathrm{Cl}(0,7)$ or Cayley–Dickson doubling across even and odd grades of $\mathrm{Cl}(3,0)$, with $\mathfrak{g}_2$ as the kernel of `Bivector.commutator(orientation)` ([octonions](../examples/math/octonions/core.py), [octonionic rolling](../examples/math/g2_rolling/core.py), [triality](../examples/math/triality/core.py)) |
| Bloch-ball states, POVM effects and Pauli transfer channels | Projective points of weight one (`Antivector-`), measurement planes (`Vector+`) paired by `effect & state`, and linear maps `Point <- Point` ([process tomography](../examples/quantum/process_tomography/process_tomography_projective.ipynb)) |
| Jones, Stokes and Mueller polarization calculus | Transverse screen slices (`ScreenSlice+`), rotor/projection maps on screen directions, and coherency correlations `Screen- <- ScreenSlice+` ([Kerr polarization](../examples/relativity/kerr/core.py)) |
| Multiparticle quantum registers and stabilizer projectors | Direct-sum signatures ($\mathrm{Cl}(3n,0)$ or $\mathrm{Cl}(3n,n)$), commuting blade involutions `(1 + blade) / 2` as stabilizer projectors, and multilinear maps across state spaces ([two spins in spacetime](../examples/quantum/two_spins/two_spins_spacetime.ipynb), [stabilizer code](../examples/quantum/stabilizer/core.py)) |
| Projective quadrics, conics and absolute metrics | Symmetric correlations `Plane+ <- Point-` and `Point- <- Plane+`, pulled back through singular projections by `.adjugate()` ([multiview reconstruction](../examples/estimation/multiview/core.py)) |
| Algebraic relationships of discrete exterior calculus | Incidence and Hodge maps on fields, dual pairings and adjoints, with chain identities such as `d1(d0) == 0` and Laplacians built by composition ([surface connections](../examples/surfaces/connections/core.py)) |
| Symplectic / Poisson brackets and Onsager–GENERIC thermodynamics | Correlations `Covector+ <- State-` (symplectic 2-form or Ruppeiner metric) and `State- <- Covector+` (Poisson bivector or Onsager mobility) on a null-metric state space ([GENERIC](../examples/thermodynamics/generic/core.py)) |
| Premetric electrodynamics and crystal optics | Constitutive correlations `Antibivector- <- Bivector+` from untwisted field 2-forms to twisted excitation 2-forms, including `Scalar-` axion couplings ([Fresnel wave surfaces](../examples/electromagnetism/fresnel/core.py)) |
| Voigt / Kelvin fourth-order elasticity and compliance | Stiffness takes strain (`Plane+ <- Point-`) to Cauchy or Piola–Kirchhoff stress (`Point- <- Plane+`); compliance maps stress back to strain. These fourth-order extensors act on second-order geometric maps ([elastic stress](../examples/mechanics/stress/core.py), [crystal waves](../examples/mechanics/crystal_waves/core.py)) |

## Numerical methods

The linear-algebra methods come with the maps. Gauss–Newton, Schur complements, conjugate gradients, least squares and generalized eigenproblems operate on geometric types from construction through solution. A camera correction is a twist that moves the camera; a vibration mode is a rate that moves the body.

The pairings and spaces remain visible throughout the algorithm. A Schur solve retains the geometric arguments of its blocks, conjugate gradients pairs a step with a cost differential, and a mode problem compares energy forms. This also helps find simpler formulations: a quadratic cost supplies its local curvature directly, and the relevant pairing can expose a symmetric energy before an artificial symmetrization changes the problem. The [common-practice comparison](common_practice.md) develops these consequences case by case.

## Closure

GA and extensors provide a unified algebra for the objects and maps of a geometric model. The inclusions establish its reach: tensors, linear maps, exterior products and forms all have their place. The geometric products relate those parts and construct the transformations acting on them.

An alternating tensor, a cross product, a reciprocity pairing and a spinor action each need their own motivation and conventions when introduced separately. Their shared construction in GA makes those relationships part of the algebra, and extensors extend them to arbitrary multilinear maps. The model can move from its objects to its laws and transformations without acquiring a separate language for each construction.

The [multiple-lenses account](multiple_lenses.md) develops the relationship between geometric construction and multilinear maps. The [orientation discussion](why_orientation.md) develops how those constructions behave under reflections.
