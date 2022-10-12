"""Realizations of the abstract algebra, multivector, and operator types in various frameworks

The only thing required to implement the abstract interface is
 * to provide a concrete backing datastructure for the multivector type
 * to provide one or more concrete implementations of the operator type
   (might sound complicated, but it really isnt)
 * provide the context object with said operator type
   (or a rule for selecting them based on the circumstances, if multiple competing solutions exist)

Note that the code provided in this module is provided more as an example implementation,
than as 'the definite way to do GA in your framework of choice'.
For instance, if you decide that your multivectors should have struct-of-arrays memory layout,
rather than an array-of-structs, or you want to use jax-dataclasses, id encourage you to just
duplicate and customize these classes in your own codebase,
which strikes me as more maintainable than making these classes configurable to be optimized for every concievable use case.

It should be fairly straightforward to add support for alternative backends that are not python based languages or frameworks,
such as HLSL, as far as code-generation for various multi-linear products is concerned.
However, the various nonlinear algorithms like norms and exponentials and what not, would require more creativity,
or reimplemention for that specific backend.
If you really want to do things properly, it would be an intriguing possibility to use JAX as a tracing / optimizing compiler,
and use the optimized JAX graphs as a starting point for code generation in other languages.
"""