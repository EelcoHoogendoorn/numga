
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
* expand test coverage
* Provide more batteries included support for common GA applications, such as conformal and projective algebras.
* make abstract operators work with sparse tensors internally, for better scaling to high dimensions
* flesh out optimized algo implementations; log/exp and the like. offer variants; precision tradeoffs; differentiability

* explore possibilities for supporting non-diagonal metrics
* expand backends (cupy, hlsl, pytorch, etc?)
* create parsed string syntax to formulate operators? ⊢, ⊣, ⊨, ⩜, ⩝
