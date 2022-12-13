The numpy backend is in a way the only 'officially supported' one,
since it is what is used to run the unit tests.

Note that my own primary use case is for compiled JAX though; 
so the numpy backend makes some concessions to JAX;
like using .at[].set syntax which adds some overhead without compilation.

That being said, if you need to do some off the cuff geometric algebra in numpy,
using numga is going to be about as fast and clean a way of doing it as can be done in numpy.