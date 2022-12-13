The JAX backend is the one I am personally most invested in;
and the overall design decision of this library will be somewhat guided 
by what is possible and desirable from the perspective of compiled JAX code.

That being said, as noted in the readme one level up, 
this is still just an example backend implementation,
and if you are wondering 'why are simple things like multivector.basic_math_operation missing?',
its simply because it hasnt been required to get the examples in this library running.

One thing that id like to improve in future version of the JAX backend, is to be able to easily choose,
between struct-of-array versus array-of-struct memory layout for both multivectors and operators,
as this can have substantial performance implications depending on the device you are running on.
