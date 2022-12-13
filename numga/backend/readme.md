The module contains realizations of the abstract algebra, 
multivector, and operator types in various frameworks

Note that the code provided in this module is provided more as an example implementation,
than as 'the definite way to do GA in your framework of choice'. 
There are various context-dependent choices to be made, 
which have no single perfect solution on the level of a library like this, 
and the level or generality we are aiming for.

For instance, if you decide that your multivectors should have struct-of-arrays memory layout,
rather than an array-of-structs, or you want to use jax-dataclasses rather than pytrees,
id encourage you to just copy and customize these classes in your own codebase,
which is more maintainable than making these classes configurable for every concievable use case.

As another illustration, using 'axis' keywords rather than 'dim' keywords in the torch backend
might horrify you if you are working in torch; but it's the right thing to do for me to be able 
to run my example code with swappable backends. Also there is some jank in the torch backend,
to make its indexing bahavior more consistent with numpy/jax, 
which you shouldnt care for if working soley in torch.

These backends are really only implemented with the goal of 
making the examples in this library runnable,
to show the general gist of things, and illustrate how simple they can be.
If you run into some missing functionality, 
do not assume it isnt there because it is difficult to add.
It probably is trivial to add; but figuring out a generic solution 
that will satisfy every possible use case, 
currently is not in the scope of this library's ambitions.

 
It should be fairly straightforward to add support for alternative backends that are not python based languages or frameworks,
such as HLSL, as far as code-generation for various multi-linear products is concerned.
However, the various nonlinear algorithms like norms and exponentials and what not, would require more creativity,
such as some custom tracing logic, or reimplemention of that logic for that specific backend.
One interesting possiblity would be to use JAX as a tracing / optimizing compiler,
and use the optimized JAX graphs as a starting point for code generation in other languages.
