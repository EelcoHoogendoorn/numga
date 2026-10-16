# Extensors through multiple lenses

The aim of this note is to provide multiple perspectives on the definition of an extensor, showing how these different readings fit together.

From computer science, extensors arise through partial application of GA products. Geometrically, an extensor built from geometric elements takes its meaning from its construction, and the tensor transformation law follows as a consequence. Mathematically, extensors are multilinear maps, equivalently expressible as sums of GA product expressions.

## 1. The computer science lens: partial application

Partial application binds some arguments of a function now and takes the rest later. A Python function can fix the light source and accept the point as its argument:

```python
def ray_fn(point: Point) -> Line:
    return light & point
```

Replacing the concrete `point` with the type `Point`, representing the space of possible points, also constructs a callable with one argument:

```python
ray = light & Point                                          # Line <- Point
```

`ray_fn` is a Python function, while `ray` is an extensor. Both are first-class Python objects, and `ray_fn(point)` and `ray(point)` return the same line.

The extensor is also a first-class element of the algebra. GA products act directly on its output, and the result still takes a point:

```python
shadow = ray ^ ground                                        # Point <- Point
```

`ray` has output type `Line`, so `^ ground` acts on that line. The resulting `shadow` has type `Point <- Point`: it still takes a point, and now returns one.

Composition supplies the output of one map to the input of another. The intersection with `ground` can itself be written as a map that takes a line:

```python
intersection = Line ^ ground                                 # Point <- Line
```

`intersection` accepts a line, and `ray` produces one. Supplying `ray` in that slot composes the two maps:

```python
shadow = intersection(ray)                                   # Point <- Point
```

The result still takes the point `ray` takes, and is the same map as `ray ^ ground`.

Evaluation is all `ray_fn` offers: a Python function hides the arithmetic it performs. An extensor is that arithmetic, held as a value, so evaluation is only one of the things it supports. It can also run in the other direction, from an output back to an input that produces it: for a linear map `M`, `M.solve(y)` finds an `x` such that `M(x)` equals `y`, and when `M` is invertible, `M.inverse()` constructs the entire inverse map, itself an extensor that can be composed and used in further products. Its adjoint, its eigenvectors and its trace are there to be taken as well; a closure gives access to none of them.

## 2. The geometric lens: meaning from construction

A tensor is considered a geometric object when it obeys the tensor transformation law, `T' = R T Rᵀ`: one factor of the rotation on each of its slots. This criterion cannot be applied to extensors, however, because the rotation map is itself an extensor: a motor is a product of planes, and its action on points, `motor >> Point`, is a map like any other. Extensors cannot be certified by a test that is itself phrased in extensors.

Nor do they need to be. The motor already has a geometric meaning, and it has it because of what it is made of: two planes, a reflection in the first followed by a reflection in the second. Any extensor built from geometric elements by the algebra's products has a geometric meaning in the same way. The inertia of a mass point is the point joined with its velocity, and its velocity under a rate of motion is its commutator with that rate:

```python
motor = second_plane * first_plane                             # Motor
inertia = point & point.commutator(Line)                       # Line <- Line
```

A motion preserves every product of the algebra, so building commutes with moving: building from moved ingredients gives the same map as building first and then composing the result with the motion. That is the tensor transformation law, reappearing as a consequence. A thin lens from the [`scenegraph` example](../examples/geometry/scenegraph/scenegraph.ipynb) is built from its centre, its plane and its focal length, and moved as a map it is the lens built anew from its moved centre and plane:

```python
lens = Line - (center & (Line ^ plane)) / focal                # Line <- Line
moved = motor >> lens(motor << Line)
rebuilt = Line - ((motor >> center) & (Line ^ (motor >> plane))) / focal
```

So nothing is given up: the tensor transformation law holds for every extensor built from geometry.

What is gained is a single account. The tensor transformation law divides the world into transformations, given in advance, and the objects it tests them against; the lens, an object and a transformation at once, shows how arbitrary that division is. Here the motor, the lens and the rays are built from the same products and moved the same way. **An extensor built from geometry transforms as its geometry does. No criterion is needed to make it geometric; the expression that builds it already reads as what it is.**

## 3. The mathematical lens: multilinear maps and sums of products

An extensor is a map between multivector spaces, linear in each of its arguments separately. The input and output spaces may contain any grades, including mixtures of grades. Its arity is the number of arguments: a linear map has one, a bilinear map has two, and a multivector itself has none.

The products of geometric algebra are bilinear. Fixing some factors and writing types for the others therefore gives a multilinear map, one argument per type; adding expressions with the same input and output spaces gives another such map. Partial application fills some slots, composition substitutes maps into slots, and filling every slot leaves a multivector. These are operations on the same family of objects.

The converse matters just as much: **every multilinear map between finite-dimensional multivector spaces can be written as a sum of expressions built from the algebra's products.** A linear map is completely determined by its outputs on a basis. Pairing the input with a suitable multivector reads one basis coefficient; multiplying that scalar by a chosen output gives a dyad.

For the point space, choose a basis and a corresponding batch of `planes`: each plane pairs to one with its own basis point and to zero with the others. The regressive product then gives one scalar readout per basis point:

```python
readouts = planes & Point                                    # [basis] Scalar <- Point
```

Assign any output point to each basis point, collected in the batch `outputs`. Multiplying those outputs by the readouts and summing constructs the map:

```python
linear = (outputs * readouts).sum(axis=0)                     # Point <- Point
```

Evaluating `linear` on a basis point selects its assigned output. Since those outputs can be chosen freely, and linearity determines every other result, this constructs every linear map from points to points.

A bilinear map assigns an output to each ordered pair of basis points. Collect these in `pair_outputs`, with one batch axis for each input basis. Each term multiplies its chosen output by one readout from each input:

```python
bilinear = (
    pair_outputs * readouts[:, None] * readouts[None, :]
).sum(axis=(0, 1))                                           # Point <- (Point, Point)
```

The two copies of `readouts` each take a point, independently. On a pair of basis points, only their assigned output survives the sum. Multilinearity determines the result for every other pair of inputs. Each further argument adds another scalar readout factor and another batch axis to sum over.

The outputs can be multivectors of any type, and the inputs can belong to any multivector space. Pairing each input grade with its complement supplies the scalar readouts; for mixed grades, the scalar part of that pairing does so. This construction works even when the metric is degenerate. Where the scalar product is nondegenerate, it can supply the readouts instead.

The multilinear-map definition and the sum-of-products construction describe the same space of extensors. The extensors of interest in practice are a subset of that space: those whose factors carry a meaning, such as a motor built from two planes or the inertia of a point built from the point and its velocity. Which constructions have a geometric meaning is a question of interpretation, not of definition; the expansion above establishes only that no multilinear map is out of reach of the products.

## How the lenses fit together

The three lenses describe one set of operations. Writing a type in place of a value in a product builds a map from geometry, filling a slot evaluates it, and filling a slot with another map composes the two. Moving a map is composition with the extensors of a transformation, and the transformation law is what that composition gives when the transformation preserves the products the map was built from. Sums of products reach every multilinear map, so nothing outside these operations is needed to build any extensor, apply it, or move it.

## Further reading

* [Extensor overview](extensors.md): applications across mechanics, optics, estimation, electromagnetism, relativity, and spin.
* [Extensor tutorial](tutorial/extensors/00-introduction.md): the mathematics developed step by step from a single open slot.
* [The adjoint](tutorial/extensors/09-the-adjoint.md) and [the adjugate](tutorial/extensors/10-the-adjugate.md): moving a map from one side of a pairing to the other.
* [Extensors, linear algebra, tensors and forms](frameworks.md) and [Extensors and common practice](common_practice.md): relationships to other mathematical frameworks and their use in applied fields.
