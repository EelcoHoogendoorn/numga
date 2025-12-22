# Extensors as partially bound geometric product expressions

This short article lays out the case for the utility of partially bound geometric product expressions (PBGPE), as first class citizens of a geometric algebra library.

At a risk of abuse of [terminology](#terminology-footnote), we will also refer to PBGPE's as 'extensors'; which at any rate is a catchier name.

In the process of laying out this case, we intend to question two common notions:
* geometric algebra is less expressive than tensor algebra
* geometric algebra forces you to choose between quaternions and matrices

# Examples

We will go over a few examples to illustrate the workings and utility of extensors.
 * [Levi-Civita symbol](#familiar-example-the-levi-civita-symbol)
 * [Complex number multiplication](#simple-example-complex-number-product-in-matrix-form)
 * [Inertia mappings](#advanced-example-inertia-mappings)
 * [Affine matrices and rotations](#performance-example-affine-matrices)

## Familiar example; the Levi-Civita symbol

To illustrate the concept of an extensor, we first turn our attention to the Levi-Civita symbol, and how it can be constructed with numga [syntax](#ref-syntax-footnote).

```python
# set up a 3d algebra context
ctx = Context((3, 0, 0))
# cons
V = ctx.subspace.vector
# Construct antisymmetric binary mapping from a pair of vectors to vectors
#  aka, the 'cross product'
e_ijk = V.wedge(V).dual()
# this is the usual antisymmetric Levi-Civita symbol
print(e_ijk.data)
[[[ 0  0  0]
  [ 0  0  1]
  [ 0 -1  0]]
 [[ 0  0 -1]
  [ 0  0  0]
  [ 1  0  0]]
 [[ 0  1  0]
  [-1  0  0]
  [ 0  0  0]]]
assert e_ijk.data.shape == (3, 3, 3)
# the cross product has two input arguments; hence arity=2
assert e_ijk.arity == 2
# as noted, all three input/output indices are over the space of vectors
assert e_ijk.axes == (V, V, V)

# construct a vector of unique integers
v = ctx.multivector.vector([1, 2, 3])
# we bind the last argument of the extensor `e_ijk`
m = e_ijk.partial({1: v})
# m is now a unary mapping from vectors to vectors
assert m.arity == 1
assert m.axes == (V, V)
assert m.data.shape == (3, 3)
# the resulting expression is 'the cross product with `v` in matrix form'
print(m.data)
[[ 0  3 -2]
 [-3  0  1]
 [ 2 -1  0]]
# Not coincidentally; this gives the same result
print(np.einsum('ijk,k->ij', e_ijk.data, v.data))
[[ 0  3 -2]
 [-3  0  1]
 [ 2 -1  0]]
```
The key realizations to be taken from the above example, are that:
 * It is in fact true in general, that we can express *any* GA product in terms of a multiplication table expressed as a sparse array of integer signs [+1, 0, -1]
 * Also, we can always view binding an argument to that expression as a simple contraction over an appropriate axis.


## Simple example; 'complex number product in matrix form':

While the Levi-Civita symbol is often viewed as a (3,3,3) shaped block of numbers, the underlying principles can be applied to any multivector product.

```python
n = 2
ctx = Context((n,0,0))
# construct the abstract space of Cl2 rotors (isomorphic to the complex numbers)
R = ctx.subspace.even_grade
# `prod` is a extensor of arity 2
prod = R * R
assert prod.arity == 2
assert prod.data.shape == (n,n,n)
# the even subalgebra is closed under multiplication
assert prod.axes == (R, R, R)
# a concrete numerical example of unique integers 
r = ctx.multivector.even_grade(np.arange(n)+1)
matrix = prod.partial({1: r})
# look ma; a matrix!
assert matrix.arity == 1
assert matrix.data.shape == (n,n)
print(matrix.data)
[[ 1  2]
 [-2  1]]
# if we bind a second argument...
result = matrix.partial({0: r})
# result is a `nullary extensor of even grade`... 
# or a complex number if you are not *that* much of a nerd
assert result.arity == 0
assert result.subspace == R
```

## Advanced example: Inertia mappings

Since numga makes extensors first class citizens of the library, extensors can also be broadcast and summed, and otherwise manipulated in every way that you would expect of a 'nullary multivector expression', or 'multivector'.

The utility thereof is nicely illustrated by the concept of inertia mappings. Since for any PGA dual vector point `p`, the instantaneous rate of change of its rigid body transform is given by the bivector `b`, the dual bivector momentum line `m` is given by:

```python
m = p.regressive(b.commutator(p))
```
<i>References: [Gunn-2011](#ref-gunn-2011), [PGADYN](#ref-pgadyn) </i>

Then it follows that the mapping from bivector rates to dual bivector momentum line of the point `p` can be expressed as:

```python
# same as above; just with delayed binding of `b`
B = b.subspace
I = p.regressive(B.commutator(p))
m = I(b)
```

The inertia mapping `I` here is an expression of arity 1; which maps from the space of bivector rates to dual bivector momentum lines. This justifies the label 'inertia map'.

In a non-GA vector-algebra context, the mapping from velocities to momenta is referred to as an inertia 'tensor'. 
Since our inertia mapping is a linear mapping from multivector to multivector, we might label it an 'extensor'

Note that momentum is a linearly additive quantity; the momentum of a collection of points, is the sum of the momenta of the individual points.
By extension, the inertia of a rigid body can be viewed as a sum over the inertia of its constituent parts.

```python
# construct 2d euclidian PGA context
n = 2
ctx = Context((n,0,1))
B = ctx.subspace.bivector
N = n + 1
# N antivector corner points of a simplex
p = ctx.multivector.vector(np.eye(N)).dual()
assert p.shape == (N,) and p.data.shape == (N, N)
# collections of mv's can be manipulated using standard numpy semantics
com = p.mean(axis=0)
assert com.shape == ()
# construct N unary inertia mappings from bivector to dual bivector;
# one for each input point `p`
Is = p.regressive(B.commutator(p))
assert Is.shape == (N,)
assert Is.data.shape == (N, len(B), len(B))
assert Is.arity == 1
# sum over the inertia contributions of all N points
I = Is.sum(axis=0)
assert I.data.shape == (len(B), len(B))
# This gives a (full rank) inertia map of the whole simplex
assert I.axes == (B.dual, B)
assert I.arity == 1
# this also 'just works', as long as the unary mapping is indeed invertible
Iinv = I.inverse()
assert Iinv.axes == (B, B.dual)
assert Iinv.arity == 1
```
Note that the example above works identically, regardless of the dimension and signature of the projective space.


## Performance example; affine matrices

Geometric algebra, or the choice to represent transformations as quaternions, or motors more generally, is often framed as being in opposition to the use of affine matrices to represent transformations instead. 

But any GA framework that provides partially bound expressions, or extensors, allows one to pick and choose the most appropriate representation for the situation at hand.

```python
ctx = Context('x+y+z+w0')
# construct a unit motor
m = ctx.multivector.even_grade()
# the subspace of motors
M = m.subspace
# the subspace of dual vectors; or points in plane-based projective algebra
P = ctx.subspace.vector.dual
# an expression of arity 3
assert M.sandwich(P).arity == 3
# this is an mv-expression of arity 1; bound over m
sandwich = m * P * m.reverse()
# same but more optimized; capable of exploiting m==m
# a sandwich with the abstract space of dual vectors
sandwich = m.sandwich(P)
# only one unbound argument, a mapping from dual-vectors to dual-vectors
assert sandwich.arity == 1
assert sandwich.axes == (P, P)
# `sandwich` is here represents 'the motor sandwich in matrix form'
assert sandwich.data.shape == (4, 4)
points = ctx.multivector(subspace=P, np.random((1_000_000, 4)))
# this dispatches a million 4x4 mat-vec multiplies 
# ( TPU go BRRRRR....)
tpoints = sandwich(points)
assert tpoints.subspace == P
assert tpoints.arity == 0
# this is the same code that will get emitted when calling `sandwich(points)`
#  it will dispatch efficient C code using a numpy backend,
#  and will produce a useful representation for the XLA compiler using the JAX backend
data = np.einsum('ij,...j->...i', sandwich.data, points.data)
assert np.allclose(tpoints.data, data)
```

Note that our GA framework does not come with an `Affine4x4` type; nor is there an explicit `motor_to_affine4x4` function, but we can still use the extensor functionality to express the same operations in a compact and performant way. Rather than viewing the affine matrix approach as being 'outside' the GA framework, it in fact becomes a first class citizen of the framework, simply by facilitating partially bound expressions as first class citizens. In addition, we are not limited to affine transformations, antivectors, or Cl(3,0,1); but any such transformations can be expressed as extensors.

# The expressivity of extensors, or PBGPE's
As a recent example of a statement that touches on the limits of the expressivity of GA, we cite the following quote from a paper investigating inertia in a PGA context:

<i>"While the scalar volume and vector center of mass have natural representations in PGA, the
inertia tensor is a rank 2 tensor that has no direct PGA equivalent." [plane-and-simplex](#ref-plane-simplex)
</i>

This is of course the case under the usual interpretation of 'PGA representations' as 'nullary extensors'. However, we have argued here that this might be an unnecessarily limiting definition of terms.

If we do expand our notion of 'PGA representations' to extensors of arbitrary arity, then it follows that the inertia tensor does have a natural representation within PGA.


# Implementation
Numga provides an implementation of extensor functionality as discussed here in the current version of the code. The implementation is fairly simple; every operator is first constructed as a sparse multiplication table; and then composed using appropriate contractions for higher arity operators. When bound with multivector arguments, we can choose to execute these as either dense contractions; or via unrolling over only the nonzero terms of the expression.

Seeing as how numga leans into the multi-staged compilation model of being a python library that sets out to trace jit compiled JAX functions, it is easy to delegate the construction of the operators to being a preprocessing step, which can easily be cached, and the execution of the operators to be jit-compiled by JAX. This allows for good JAX performance in a relatively simple library. 

How to implement extensors as such in a language without granular control over compilation stages, such as a more 'traditional' C++ metaprogramming framework, is an open question. 

## Example code
For illustration, we here work out how a ternary rotor-vector sandwich operation works under the hood, in combined numpy+JAX code.

```python
ctx = Context('x+y+')
# the subspace of even grade multivectors; isomorphic to complex numbers
Q = ctx.subspace.even_grade()
# the subspace of vectors
V = ctx.subspace.vector

# Lets examine how the Cl2 rotor-vector sandwich comes to be 
#  output = q * v * q.reverse()
sandwich = Q.sandwich(V)
assert sandwich.data.shape == (2,2,2,2)
assert sandwich.axes == (V, Q, V, Q)
print(sandwich.data)
[[[[ 1  0]
   [ 0 -1]]
  [[ 0  1]
   [ 1  0]]]
 [[[ 0 -1]
   [-1  0]]
  [[ 1  0]
   [ 0 -1]]]]

reverse = Q.reverse()
# reversing a simple rotor is just negating its bivector part
# this just follows the standard GA logic of counting the number of swaps required to map the reversed basis vectors to their original positions
# We here resist the urge to prematurely optimize this representation; but rather lean into the general form of a linear map to represent the reverse at this stage, so as to keep the code for combining it with other extensors simple.
print(reverse.data)
[[ 1  0]
 [ 0 -1]]

# left hand side of the sandwich;
# the product of Q and V produces another V
# This multiplication table is again constructed using the standard GA logic; 
# eliminating repeating terms that contract via the metric; and then mapping to a standardized ordering of basis blades
# We may symbolically deduce in this step that the output space is V; 
# these being the only nonzero terms that emerge from the multiplication table
left = Q.product(V)
assert left.data.shape == (2,2,2)
assert left.axes == (V, Q, V)
print(left.data)
[[[ 1  0]
  [ 0  1]]
 [[ 0 -1]
  [ 1  0]]]

# right hand side of the sandwich; (Q * V) * Q.reverse()
right = V.product(Q)
assert right.data.shape == (2,2,2)
assert right.axes == (V, V, Q)

# we can bind the reverse extensor to the Q input slot of the right side product extensor to get the combined extensor
right_reverse = np.einsum('ijk, kl -> ijl', right.data, reverse.data)

# Now we may bind the output of right_reverse to the V input slot of left Q*V product 
sandwich = np.einsum('ijk, klm -> ijlm', left.data, right_reverse)
# we now have a 4d array representing the ternary sandwich operator
assert sandwich.shape == (2,2,2,2)
# Since we are merely explicitly retracing the same steps numga takes under the hood,
# we obtain the same result as we get by letting numga handle the binding of arguments
print(sandwich)
[[[[ 1  0]
   [ 0 -1]]
  [[ 0  1]
   [ 1  0]]]
 [[[ 0 -1]
   [-1  0]]
  [[ 1  0]
   [ 0 -1]]]]

# The below is what happens when binding a specific rotor q to the sandwich extensor
# note that the (2,2,2,2) sandwich kernel is not an argument of the sandwich_map function
# rather it is a compile time constant for the purpose of this jitting context.
# It does not depend on the numerical values of the rotor, but only its type/subspace,
# and is computed only once in a cached manner.
@jax.jit
def sandwich_map_vector(r):
    # given a rotor r as a shape (2,) array, return a (2,2) matrix 
    # representing the sandwich operation with a vector space V
    return jnp.einsum('ijkl, j, l -> ik', sandwich, r, r)

def explicit_unroll(r):
    # since `sandwich` is a compile time constant, we can unroll over the nonzero terms
    o = np.empty((2,2))
    o[0,0] = +r[0] * r[0] - r[1] * r[1]
    o[0,1] = +r[0] * r[1] + r[0] * r[1]
    o[1,0] = -r[0] * r[1] - r[0] * r[1]
    o[1,1] = +r[0] * r[0] - r[1] * r[1]
    return o

def explicit_unroll_opt(r):
    # after simple term rewriting we may expect of the XLA compiler toolchain, 
    # we should end up with code similar to the following
    o = np.empty((2,2))
    d = r[0] * r[0] - r[1] * r[1]
    od = 2 * r[0] * r[1]
    o[0,0] = d
    o[0,1] = od
    o[1,0] = -od
    o[1,1] = d
    return o
```


## Sparsity

Note that we can elect to execute the binding of arguments either via contractions over dense arrays; or via unrolling over only the nonzero terms of the expression. This is supported in numga 1.0. 

Additionally, we might choose to *store* the multiplication table of the extensor in a sparse format as well. Currently, only dense storage is supported; it keeps the implementation simple, and in modest dimensional algebras (<6d) this has not been a problem. Some of the worst case constructions one might encounter in practice are the construction of a versor sandwich in 5d with a trivector, representing a CGA point pair. The versor would have 16 components, and the trivector 10; prior to binding any arguments to the ternary operation, the resulting multiplication table would be of shape (16,10,16,10), or 25.6kb. Again, note that such an object would only be constructed outside the scope of an innermost jit operation, and is thus a compile time constant. Fetching such an object from RAM should take a mere fraction of a microsecond, and manipulation with dense einsum manipulations is fast enough. However, one can easily see how working with such dense logic can result in compilations times that get out of hand, with more exotic higher dimensional algebras.

Such a sandwich operation has 'only' 1600 nonzero terms that would be required to unroll it; either to transform a single trivector, or to bind only the versor and accumulate the coefficients of the matrix. Note that in the Cl(3,0,1) case, performing a well optimized versor-point sandwich directly can be [competitive](#ref-look-ma) with a 4x4 matrix multiplication. However, in higher dimensions, a conversion to a matrix form will pay off quickly, when there is more than one object to transform; as we can see from the 1600 nonzero terms in the sandwich expression, versus a mere 10x10=100 fused-multiply-adds in the matrix form.


## Syntax footnote
The syntax presented here is partially aspirational; the current numga syntax 1.0 does not allow inline mixing of concrete multivector types and abstract subspaces. Rather, it requires an explicit syntax for constructing operators with unbound arguments. The syntax as presented here is as-planned for numga 2.0. 

Related to this, numga 1.0 has both a 'MultiVector' type as well as an 'Operator' type; but from the numga 2.0 / extensor perspective, this is merely an arbitrary distinction between nullary and non-nullary extensors. This shows itself in an unwanted code duplication between the two types, as it pertains to broadcasting semantics and other generic functionality.

## Performance footnote
With regards to performance of implementing extensor functionality; while there is an additional `if` statement in each operator invocation required to differentiate concrete multivector arguments, from operations to be constructed over an abstract subspace; these are compile-time if-statements from the perspective of a JIT-compiled expression (whether JAX, torch or some other compilable backend), which is the setting numga concerns itself with. Using the non-compiled numpy backend, enabling expressive extensor syntax does carry this overhead; but the numpy backend is really only there for unit testing and educational purposes, and achieving optimal performance there is considered a non-goal of numga, and one more python conditional is not going to make a material difference.

## Terminology footnote

The label 'extensor' is first claimed by [Hestenes](#ref-ca-to-gc), who defines it simply as any multilinear function of multivector arguments.

This defining reference contains several reflections on historical competing ways of defining 'tensors', to which we do not have anything new to add. We note that 'tensors' as multilinear functions of 1-vectors are strictly a subset of the 'extensors'. 

As Hestenes also qualifies subsequently, and is well known, the above definition is incomplete; not every n+1-dimensional block of random numbers defines a valid n-ary extensor, any more than every block of random numbers would define a valid tensor, in the sense of obeying frame-invariance. 

However, any extensor built up from valid operations on multivectors within the algebra is a valid extensor in terms of transformation properties (and also self-evidently a multi-linear map over multivectors). Hence we emphasize this definition as the 'extensors' of numga; as this is the only type of object that we are dealing with, and this definition self-evidently falls within the initial definitions of Hestenes (as well as subsequent clarifications).

Note that this definition of 'extensor', does not include any notion of contravariant or covariant indices; or separable metric tensors. While one could model a metric tensor as an extensor, it would be an object modelled with the library; not a first class concern of the library or its extensor concept itself. The choice to see the metric as intrinsic to the algebra, rather than as separate from it, is rather fundamental to the viewpoint difference in geometric algebra versus exterior algabra as a separate subject. The above reference by Hestenes also goes into this aspect in some depth, for those interested. 

Put more plainly, if you are looking to rotate vectors or build inertia matrices, the numga extensor concept is probably exactly what you are looking for. If you are looking to model the Einstein field equations... maybe numga will be of help to you, but if you have to ask... you should probably first read all the literature on gauge-theory-gravity (GTG), which is more than the authors of numga can claim to have read.


## References
* <a id="ref-gunn-2011"></a>**[Gunn-2011]** Gunn, C. (2011). *Geometry, Kinematics, and Rigid Body Mechanics in Cayley-Klein Geometries*. [Link](https://www.researchgate.net/publication/265672134_Geometry_Kinematics_and_Rigid_Body_Mechanics_in_Cayley-Klein_Geometries)
* <a id="ref-pgadyn"></a>**[PGADYN]** *PGA Dynamics*. [Link](https://bivector.net/PGADYN.html)
* <a id="ref-look-ma"></a>**[look-ma-no-matrices]** *Look Ma, No Matrices!* [Link](https://enkimute.github.io/LookMaNoMatrices/)
* <a id="ref-plane-simplex"></a>**[plane-and-simplex]** *Clean up your Mesh: Plane and Simplex*. [Link](https://www.researchgate.net/publication/397490460_Clean_up_your_Mesh_Plane_and_simplex)
* <a id="ref-ca-to-gc"></a>**[ca-to-gc]** *Clifford Algebra to Geometric Calculus*. [Link](https://www.researchgate.net/publication/258944244_Clifford_Algebra_to_Geometric_Calculus_A_Unified_Language_for_Mathematics_and_Physics)
