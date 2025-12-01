# Extensors as partially bound geometric product expressions

This short article lays out the case for the utility of partially bound geometric product expressions (PBGPE), as first class citizens of a geometric algebra library.

At a risk of abuse of terminology, we will also refer to PBGPE's as 'extensors'; which at any rate is a catchier name.

In the process of laying out this case, we hope to expose two common notions as misconceptions:
* geometric algebra is less expressive than tensor algebra
* geometric algebra forces you to choose between quaternions and matrices

We will go over a few examples to illustrate the workings and utility of extensors.
 * [Levi-Civita symbol](#familiar-example-the-levi-civita-symbol)
 * [Complex number multiplication](#simple-example-complex-number-product-in-matrix-form)
 * [Inertia mappings](#advanced-example-inertia-mappings)
 * [Affine matrices](#performance-example-affine-matrices)

## Familiar example; the Levi-Civita symbol

To illustrate the concept of an extensor, we first turn our attention to the Levi-Civita symbol, and how it can be constructed with numga syntax [1](#syntax-footnote).

```python
# set up a 3d algebra context
ctx = Context((3,0,0))
V = ctx.subspace.vector
# Construct antisymmetric binary mapping from a pair of vectors to vectors
#  aka, the 'cross product'
e_ijk = (V ^ V).dual()
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
assert e_ijk.data.shape == (3,3,3)
# the cross product has two input arguments; hence arity=2
assert e_ijk.arity == 2
# as noted, all three input/output indices are over the space of vectors
assert e_ijk.axes == (V, V, V)

# construct a vector of unique integers
v = ctx.multivector.vector([1, 2, 3])
# we bind one of the arguments of the extensor `e_ijk`
m = e_ijk.partial({0: v})
# m is now a unary mapping from vectors to vectors
assert m.arity == 1
assert m.axes == (V, V)
assert m.data.shape == (3, 3)
# the resulting expression is 'the cross product with `v` in matrix form'
print(m.data)
[[ 0  3 -2]
 [-3  0  1]
 [ 2 -1  0]]
# Not coicidentally; this gives the same result
print(np.einsum('ijk,k->ij', e_ijk.data, v.data))
[[ 0  3 -2]
 [-3  0  1]
 [ 2 -1  0]]
```
The key realizations to be taken from the above example, are that:
 * It is infact true in general, that we can express *any* GA product in terms of a multiplication table expressed as a sparse array of integer signs [+1, 0, -1]
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
# This give a (full rank) inertia map of the whole simplex
assert I.axes == (B.dual, B)
assert I.arity == 1
# this also 'just works', as long as the unary mapping is indeed invertable
Iinv = I.inverse()
assert Iinv.axes == (B, B.dual)
assert Iinv.arity == 1
```
Note that the example above works identically, regardless of the dimension and signature of the projective space.


## Performance example; affine matrices

Geometric algebra, or the choice to represent transformations as quaterions, or motors more generally, is often framed as being in opposition to the use of affine matrices to represent transformations instead. 

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
```

Note that our GA framework does not come with an `Affine4x4` type; nor is there an explicit `motor_to_affine4x4` function, but we can still use the extensor functionality to express the same operations in a compact and performant way. Rather than viewing the affine matrix approach as being 'outside' the GA framework, it in fact becomes a first class citizen of the framework, simply by facilitating partially bound expressions as first class citizens. In addition, we are not limited to affine transformations, antivectors, or Cl(3,0,1); but any such transformations can be expressed as extensors.


# Implementation
Numga provides an implementation of such extensor functionality in its current form. The implementation is fairly simple; every operator is first constructed as a sparse multiplication table; and then composed using appropriate contractions for higher arity operators. When bound with multivector arguments, we can choose to execute these as either dense contractions; or via unrolling over only the nonzero terms of the expression.

Seeing as how numga leans into the multi-staged compilation model of being a python library that sets out to trace jit compiled JAX functions, it is easy to delegate the construction of the operators to being a preprocessing step, which can easily be cached, and the execution of the operators to be jit-compiled by JAX. This allows for good JAX performance in a relatively simple library. 

How to implement extensors as such in a language without granular control over compilation stages, such as a more 'traditional' C++ metaprogramming framework, is an open question. 


## Syntax footnote
The syntax presented here is partially aspirational; the current numga syntax 1.0 does not allow inline mixing of concrete multivector types and abstract subspaces. Rather, it requires an explicit syntax for constructing operators with unbound arguments The syntax as presented here as is as-planned for numga 2.0. 

Related to this, numga 1.0 has both a 'MultiVector' type as well as an 'Operator' type; but from the numga 2.0 / extensor perspective, this is merely an arbitrary distinction between nullary and non-nullary extensors. This shows itself in an unwanted code duplication between the two types, as it pertains to broadcasting semantics and other generic functionality.

## Performance footnote
With regards to performance of implementing extensor functionality; while there is an additional `if` statement in each operator invocation required to differentiate concrete multivector arguments, from operations to be constructed over an abstract subspace; these are compile-time if-statements from the perspective of a JIT-compiled expression (wether JAX, torch or some other compilable backend), which is the setting numga concerns itself with. Using the non-compiled numpy backend, enabling expressive extensor syntax does carry this overhead; but the numpy backend is really only there for unit testing and educational purposes, and achieving optimal performance there is considered a non-goal of numga, and one more python conditional is not going to make a material difference.