# Extensors through multiple lenses

This document sets out three ways to read an extensor and shows that they describe the same objects.

From computer science, extensors arise through partial application of the products of geometric algebra. Geometrically, an extensor built from elements of the algebra takes its meaning and its transformation behaviour from its construction. Mathematically, extensors are multilinear maps between multivector spaces, and every such map is a sum of expressions built from the algebra's products.

## 1. The computer science lens

Partial application binds some arguments of a function now and takes the rest later. In projective three-space, planes are extrinsic vectors (`Vector+`), joined lines are intrinsic antibivectors (`Antibivector-`), and points are intrinsic antivectors (`Antivector-`), with `+` and `-` marking how each transforms in a mirror ([Why orientation](orientation.md)). Fixing a `light` source, a Python function can accept a `point` and return the line joining them:

```python
def ray_fn(point: Point) -> Line:
    return light & point
```

Writing the type `Point` for the argument, naming the space of points it takes, constructs a map directly in the algebra:

```python
ray = light & Point                                          # [] Antibivector- <- Antivector-
```

`ray_fn` is a Python function, while `ray` is an extensor. Both are first-class Python objects, and `ray_fn(point)` and `ray(point)` return the same line.

Because `ray` belongs to the algebra, the algebra's products act directly on its output, and the result still takes a point:

```python
shadow = ray ^ ground                                        # [] Antivector- <- Antivector-
```

`ray` returns a line, so `^ ground` meets that line with the ground plane. The resulting `shadow` still takes a point, and now returns one.

Composition supplies the output of one map to the input of another. Meeting a line with `ground` can itself be written as a map that takes a line:

```python
intersection = Line ^ ground                                 # [] Antivector- <- Antibivector-
```

`intersection` accepts a line, and `ray` produces one. Supplying `ray` as that input composes the two maps:

```python
shadow = intersection(ray)                                   # [] Antivector- <- Antivector-
```

The result still takes the point `ray` takes, and is the same map as `ray ^ ground`.

Evaluation is all `ray_fn` offers: a Python function hides the arithmetic it performs. An extensor holds that arithmetic as a value, so evaluation is only one of the operations it supports. It can also run in the other direction, from an output back to an input that produces it: for a linear map `M`, `M.solve(y)` finds an `x` such that `M(x) == y`, and when `M` is invertible, `M.inverse()` constructs the inverse map, itself an extensor that can be composed and used in further products. Its adjoint, its adjugate, its eigenvectors and its trace are available in the same way; a Python function gives access to none of them.

## 2. The geometric lens

An extensor built from geometry transforms as its geometry does: building it again from its transformed elements gives the same map as transforming the map itself, on its output and on each of its inputs. That holds because a motion or reflection of space is itself an element of the algebra, and its action on an element, `transformation >> x`, preserves every product.

Because those transformations and their actions are themselves elements and maps of the algebra, the same map can act as a transformation in one expression and be transformed as an object in another. A thin lens in a [scenegraph](../examples/geometry/scenegraph/scenegraph.ipynb) example bends each incoming ray toward its optical `center` where the ray meets its `plane`, in proportion to `1 / focal`. At the same time, the lens is an object in the scene, carried by a rigid motion `motor` along with everything else. Moving the lens as a map, by undoing the motion on the incoming ray, applying the lens and carrying the outgoing ray forward, gives the same map as building the lens from its moved centre and plane:

```python
lens = Line - (center & (Line ^ plane)) / focal                          # [] Antibivector- <- Antibivector-
moved = motor >> lens(motor << Line)                                     # [] Antibivector- <- Antibivector-
rebuilt = Line - ((motor >> center) & (Line ^ (motor >> plane))) / focal # [] Antibivector- <- Antibivector-
```

Because `motor >>` distributes across every product and cancels `motor <<` on `Line`, `moved` and `rebuilt` are the same map. One action moves the ray, the lens and the motion itself, and nothing in the scene is set apart from the objects it transforms.

<p align="center"><img src="../plots/scenegraph.gif" alt="A robot arm in 3D with a two-lens camera and sample sight ray on the left, and the rendered sensor image on the right" width="598" /></p>

*Figure: an articulated robot arm and a moving two-lens camera in three dimensions on the left, with the image produced on the sensor by composing the body motors, lens maps and sensor meet on the right.*

In tensor algebra the order is the reverse. The transformations come first, given as rotation matrices, and a tensor is defined as an array that responds to them by the tensor transformation law, $T' = R T R^{\mathsf T}$, one factor of the rotation on each of its axes. The rotation is a linear map on vectors and therefore a tensor itself, yet the definition treats it as given beforehand, standing outside as the test the other objects must pass. Here a transformation is an extensor from the start, and no object is defined by testing it against something posited outside the algebra. The tensor transformation law is a consequence instead: the expression for the moved lens, `motor >> lens(motor << Line)`, reads in that notation as $R T R^{\mathsf T}$, the motion applied on the output and undone on the input.

## 3. The mathematical lens

An extensor is a map between multivector spaces, linear in each of its arguments separately. The input and output spaces may contain any grades, including mixtures of grades. Its arity is the number of arguments: a linear map has one, a bilinear map has two, and a multivector itself has none.

The products of geometric algebra are bilinear. Fixing some factors and naming the spaces of the remaining arguments by their types therefore gives a multilinear map, one argument per type, and adding expressions with the same input and output spaces gives another such map. Partial application supplies some arguments, composition feeds maps into arguments, and supplying every argument produces a multivector. All of these operate on one family of objects.

The converse matters just as much: every multilinear map between finite-dimensional multivector spaces can be written as a sum of expressions built from the algebra's products. A linear map is completely determined by its outputs on a basis. Pairing the input with a suitable multivector reads one basis coefficient; multiplying that scalar by a chosen output gives a dyad.

Given a basis of the point space and the dual batch of `planes`, each plane pairs to one with its own basis point and to zero with the others. The regressive product then gives one scalar readout per basis point:

```python
readouts = planes & Point                                    # [basis] Scalar+ <- Antivector-
```

With a batch `outputs` holding the output point chosen for each basis point, multiplying by the readouts and summing constructs the map:

```python
linear = (outputs * readouts).sum(axis=0)                    # [] Antivector- <- Antivector-
```

Evaluating `linear` on a basis point selects its assigned output. Because those outputs can be chosen freely and linearity fixes every other value, this expression constructs every linear map from points to points.

A bilinear map assigns an output to each ordered pair of basis points. With those outputs in `pair_outputs`, indexed by one batch axis for each input basis, each term multiplies its output by one readout on each input:

```python
bilinear = (
    pair_outputs * readouts[:, None] * readouts[None, :]
).sum(axis=(0, 1))                                           # [] Antivector- <- (Antivector-, Antivector-)
```

The two copies of `readouts` each take a point, independently. On a pair of basis points, only their assigned output survives the sum, and multilinearity fixes the result for every other pair of inputs. Each further argument adds another scalar readout factor and another batch axis to sum over.

The outputs can be multivectors of any type, and the inputs can belong to any multivector space. Pairing each input grade with its complement supplies the scalar readouts; for mixed grades, the scalar part of that pairing does so. This construction works even when the metric is degenerate. Where the scalar product is nondegenerate, it can supply the readouts instead.

The multilinear-map definition and the sum-of-products construction describe the same space of extensors. The extensors of interest in practice are a subset of that space: those whose factors carry a meaning, such as a motor built from two planes or a lens built from its centre and plane. Which constructions have a geometric meaning is a question of interpretation, not of definition; the expansion above establishes only that no multilinear map is out of reach of the products.

In linear algebra, pulling linear forms back across the canonical pairing defines the dual map on dual spaces, whereas transferring vectors across an inner product defines the adjoint [[1, §§3F, 7A]](#references); the adjugate across `&` and the adjoint across the scalar product make the same separation here. Standard presentations of linear algebra within geometric algebra start from linear maps on vectors and extend them over outer products as grade-preserving outermorphisms [[2, §2.3]](#references), whereas the extensors considered here include multilinear maps of any arity between arbitrary oriented and mixed-grade spaces.

## How the lenses fit together

The three lenses describe one set of operations. Naming an argument's space by its type in a product builds a map from geometry, supplying a value evaluates it, and supplying another map composes the two. Moving a map is composition with a transformation's own action, and the tensor transformation law is what that composition gives because the transformation preserves the products the map was built from. Sums of products reach every multilinear map, so nothing outside the algebra's products is needed to build any extensor, apply it, or move it.

## Companion documents

* [Why orientation](orientation.md): why geometric constructions and positive physical energies require intrinsic and extrinsic orientation.
* [The regressive product](regressive_product.md): how `&` pairs orientations, joins points, contracts fields and builds linear maps and their adjugates.
* [Complementarity](complementarity.md), [Extensors, linear algebra, tensors and forms](frameworks.md) and [Extensors and common practice](common_practice.md): how these maps pair complementary quantities, relate to other mathematical frameworks, and apply across geometry and physics.
* `numga` documentation: [Extensor overview](../docs/extensors.md), [Extensor tutorial](../docs/tutorial/extensors/00-introduction.md), [The adjoint](../docs/tutorial/extensors/09-the-adjoint.md) and [the adjugate](../docs/tutorial/extensors/10-the-adjugate.md).

## References

1. [*Linear Algebra Done Right*](https://linear.axler.net/LADR4e.pdf) (Sheldon Axler, 4th ed., 2024).
2. [*Gravity, Gauge Theories and Geometric Algebra*](https://arxiv.org/pdf/gr-qc/0405033) (Anthony Lasenby, Chris Doran and Stephen Gull, *Philosophical Transactions of the Royal Society A*, 1998; arXiv, 2004).
