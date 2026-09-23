
This is a non-ordered, tentative list of aspects of this library that may recieve attention in the future. 
Or potentially, it may serve as inspiration for people who feel inclined to contribute to this library.

* set up proper packaging (pip and conda/conda forge)
* set up proper CI


* extend type system; ga-type, encapsulating norm-status, simple-lines, or other aspects of values? primal/dual flag?
* reconsider subspace implementation. is bitfield superior? or should we allow for differentiated blade order? just an added sign field?

* unify partial application of operators and binding; make sure operator kernels can also broadcast. concrete-operator as multivector, and symbolic-operator as subspace
* should we add support for anti-norms, degenerate/weight norms, and the like?
* add nice benchmarking suite on real world workflows
* expand section of examples
  * possible: diagonalize an arbitrary physical 3D inertia tensor using the extensor eigensolve. Extract principal moments and axes, construct the rotor into the principal frame, and show the inertia map becoming diagonal in that frame.
  * completed: `rewrite/examples/relativity/curvature.py` demonstrates a nonzero, square-zero vacuum wave curvature extensor and its observer-bound tidal response. Static PNG and animated GIF show the null-plane mapping and plus/cross/circular detector rings.
  * completed: `rewrite/benchmarks/motor_map.py` has focused scenarios for JAX map amortization, the 4D-vector/CGA-trivector unrolling crossover, and NumPy's unrolling cost. Each prints a small comparison and explains its measured result in its docstring.
  * possible: a small scene graph with rotations, translations, and nonuniform scales. Compose each hierarchy path into one world extensor, including the shear introduced by interleaved rotations and scales, then apply it once to the mesh. Show how rigid and non-rigid transforms share the same composition machinery and how per-vertex work becomes independent of hierarchy depth.
* expand test coverage
* Provide more batteries included support for common GA applications, such as conformal and projective algebras.
* make abstract operators work with sparse tensors internally, for better scaling to high dimensions
* flesh out optimized algo implementations; log/exp and the like. offer variants; precision tradeoffs; differentiability

* explore possibilities for supporting non-diagonal metrics
* expand backends (cupy, hlsl, pytorch, etc?)
* create parsed string syntax to formulate operators? ⊢, ⊣, ⊨, ⩜, ⩝

numga 2.0
* have byte-string-blades flyweight types, to encoder arbitrary basis blade permutations
  * byte-string-blade can construct sorted bit-blade with sign bit
  * have algebras with canonical byte-string-blades; preferential sorting of 3d pga and so on
  * allows for casting multivectors into a preferred basis sorting
* unify multivector and operator types into single extensor type
  * multivectors as nullary extensors
  * can we view a subspace as a unit unary extensor? something bound with a multivector becomes that multivector. I think that this is true by definition. but if an extensor has-a tuple of subspaces named axes; can there be an is-a relationship between the extensor and the subspace too?
  * should extensors be a type; or an interface?
  * change order of operator axes; make output axis first axis
  * make it so any extensor of valid output type can bind into any extensor input argument
