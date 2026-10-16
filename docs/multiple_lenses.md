# Extensors through multiple lenses

The aim of this note is to provide multiple perspectives on the definition of an extensor, showing how these different readings fit together.

From computer science, extensors arise through partial application of GA products. Geometrically, objects and transformations inhabit the same algebra, and transforming a map is composition. Mathematically, extensors are multilinear maps, equivalently expressible as sums of GA product expressions.

## 1. The computer science lens: partial application

Partial application binds some arguments of a function and leaves the remaining arguments open. A Python function can fix the light source and accept the point as its argument:

```python
def ray_fn(point: Point) -> Line:
    return light & point
```

Replacing the concrete `point` with the type `Point`, representing the space of possible points, also constructs a callable with one argument:

```python
ray = light & Point                                          # Line <- Point
```

`ray_fn` is a Python function, while `ray` is an extensor. Both are first-class Python objects, and `ray_fn(point)` and `ray(point)` return the same line.

The extensor is also a first-class element of the algebra. GA products act directly on its output while its point argument stays open:

```python
shadow = ray ^ ground                                        # Point <- Point
```

`ray` has output type `Line`, so `^ ground` acts on that line. The resulting `shadow` has type `Point <- Point`: its output is a point, and its point argument is still open.

Composition supplies the output of one map to the input of another. The intersection with `ground` can itself be written as a map with an open line argument:

```python
intersection = Line ^ ground                                 # Point <- Line
```

`intersection` accepts a line, and `ray` produces one. Supplying `ray` in that slot composes the two maps:

```python
shadow = intersection(ray)                                   # Point <- Point
```

The point argument of `ray` remains open, giving the same map as `ray ^ ground`.

Evaluation is all `ray_fn` offers: a Python function hides the arithmetic it performs. An extensor is that arithmetic, held as a value, so evaluation is only one of the things it supports. It can also run in the other direction, from an output back to an input that produces it: for a linear map `M`, `M.solve(y)` finds an `x` such that `M(x)` equals `y`, and when `M` is invertible, `M.inverse()` constructs the entire inverse map, itself an extensor that can be composed and used in further products. Its adjoint, its eigenvectors and its trace are there to be taken as well; a closure gives access to none of them.

## 2. The geometric and physical lens: objects and transformations in one algebra

Geometric algebra expresses transformations through its own elements. A motor is the geometric product of an even number of vectors. It acts on a point through the sandwich product and can itself be transformed by another motor. Leaving the point open, `motor >> Point`, makes that action an extensor. **The objects, the maps acting on them, and the transformations that move those maps are constructed within the same algebra.**

In the [`scenegraph` example](../examples/geometry/scenegraph/scenegraph.ipynb), a thin lens's center, plane, and focal length specify its location and action on rays. Two such lenses, a pupil, and a sensor compose into a camera:

```python
front_lens = Line - (center & (Line ^ plane)) / focal_front    # Line <- Line
rear_local = Line - (center & (Line ^ plane)) / focal_rear    # Line <- Line
rear_lens = rear_placement >> rear_local(rear_placement << Line)

to_sensor = Line ^ sensor_plane                              # Point <- Line
camera = to_sensor(rear_lens(front_lens(Point & pupil)))       # Point <- Point
world_camera = camera(camera_pose << Point)                  # Point <- Point
```

Each lens is treated as an object when it is built and placed, and as a transformation when it is applied. Their composition is a camera, which can in turn be placed and applied. The extensor approach blurs the usual distinction between object and map: the same extensor can be transformed as an object and applied as a map.

Moving a lens must preserve its relationship with the rays: if the lens and an incoming ray undergo the same rigid displacement, the outgoing ray must undergo that displacement too. This determines the moved map. Undo the placement on the incoming ray, apply the local lens, and carry the outgoing ray forward again. That is the construction `rear_placement >> rear_local(rear_placement << Line)`.

The placement action `rear_placement >> Line` is itself an extensor, built by leaving an argument open in a product. **Moving a map as an object is composing it with other maps.** With several input slots, the placement is undone on each input and carried forward on the output. The algebra supplies both the map being moved and the transformations doing the moving.

The camera's final placement changes only its input because its output stays in the camera's own sensor frame: `world_camera` takes world points and returns local sensor points. Which slots move follows from what those slots represent.

The lens's geometric construction gives a second route to the moved map: move its center and plane, then rebuild it. For a rigid placement `motor`, these two expressions give the same operator:

```python
rebuilt_lens = Line - ((motor >> center) & (Line ^ (motor >> plane))) / focal_front
moved_lens = motor >> front_lens(motor << Line)
```

Rotations and translations preserve the products used here. Moving the ingredients through the same products therefore moves the result, which is why `rebuilt_lens` and `moved_lens` agree. This argument applies to any construction whose products are preserved by the transformation: the geometry of its ingredients determines how the completed map transforms.

**The transformation law follows from the construction.** Composition tells us how to move a map, and preservation of the products makes that agree with moving the geometry it was built from. The law expresses a relationship between objects already available in the algebra.

A tensor is classically defined as something that transforms the right way, and checking that a quantity does is a standard exercise. Here the question does not arise. The map is built from geometric elements, the transformations are built from the same algebra, and how the one moves under the other follows from their products and composition: under any transformation that preserves those products, a map built from geometry moves with its geometry, and cannot move any other way. The transformations are extensors themselves, so a definition by transformation law would presuppose the very kind of object it sets out to define. A construction presupposes only the algebra.

## 3. The mathematical lens: multilinear maps and sums of products

An extensor is a map between multivector spaces, linear in each of its arguments separately. The input and output spaces may contain any grades, including mixtures of grades. Its arity is the number of arguments: a linear map has one, a bilinear map has two, and a multivector itself has none.

The products of geometric algebra are bilinear. Fixing some factors and leaving others independently open therefore gives a multilinear map; adding expressions with the same input and output spaces gives another such map. Partial application fills some slots, composition substitutes maps into slots, and filling every slot leaves a multivector. These are operations on the same family of objects.

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

The two copies of `readouts` have independent open point arguments. On a pair of basis points, only their assigned output survives the sum. Multilinearity determines the result for every other pair of inputs. Each further argument adds another scalar readout factor and another batch axis to sum over.

The outputs can be multivectors of any type, and the inputs can belong to any multivector space. Pairing each input grade with its complement supplies the scalar readouts; for mixed grades, the scalar part of that pairing does so. This construction works even when the metric is degenerate. Where the scalar product is nondegenerate, it can supply the readouts instead.

The multilinear-map definition and the sum-of-products construction describe the same space of extensors. A product expression need not be a basis expansion: its factors and products can directly express the geometric relationships that define the map. The expansion establishes that this way of constructing maps has full generality.

## How the lenses fit together

The three lenses describe one set of operations. Leaving a slot open in a product builds a map from geometry, filling a slot evaluates it, and filling a slot with another map composes the two. Moving a map is composition with the extensors of a transformation, and the transformation law is what that composition gives when the transformation preserves the products the map was built from. Sums of products reach every multilinear map, so nothing outside these operations is needed to build any extensor, apply it, or move it.

## Further reading

* [Extensor overview](extensors.md): applications across mechanics, optics, estimation, electromagnetism, relativity, and spin.
* [Extensor tutorial](tutorial/extensors/00-introduction.md): the mathematics developed step by step from a single open slot.
* [The adjoint](tutorial/extensors/09-the-adjoint.md) and [the adjugate](tutorial/extensors/10-the-adjugate.md): moving a map from one side of a pairing to the other.
* [Extensors, linear algebra, tensors and forms](frameworks.md) and [Extensors and common practice](common_practice.md): relationships to other mathematical frameworks and their use in applied fields.
