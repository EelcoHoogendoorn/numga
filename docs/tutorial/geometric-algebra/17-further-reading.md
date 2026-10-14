# 17. Further reading

This tutorial has explored the consequences of relaxing one of the rules that define the scalar product: two different basis vectors no longer multiply to zero, and only their products in the two orders cancel. That single change gives rise to a much richer algebra, in which turns, reflections, areas, volumes and intersections all follow from one product. The choice may look arbitrary at first, but its utility proves itself in its applications, from the geometry of a scene to the physics of spinning particles.

The tutorial is an entry point, not an exhaustive treatment. It covers the operations that geometric constructions are built from, and leaves the constructions themselves to the literature. Natural topics to continue with are:

1. The exponential of a bivector, which turns by any angle in the bivector's plane, much as the exponential of an imaginary number turns in the complex plane.
2. The sandwich product, which multiplies an element by a versor on one side and by its inverse on the other, and so turns, reflects or moves points, lines, planes and every other element in the same way.
3. Projective geometric algebra, which adds a basis vector that squares to zero, so that points, lines and planes anywhere in space, and the rigid motions between them, are elements of the algebra.
4. Conformal geometric algebra, which adds two basis vectors, so that circles and spheres are elements as well.
5. Geometric calculus, in which the derivative is a vector, and its geometric product with a field gives the divergence and the curl together.
6. Extensors, linear maps written in the algebra, which the [extensor tutorial](../extensors/00-introduction.md) takes up from here.

Applications run from geometry and mechanics to relativity and quantum physics. A few runnable examples that stay close to the algebra itself:

- [Two mirrors](../../../examples/math/reflections/kaleidoscope.ipynb), whose product is a turn by twice the angle between them.
- [A ball rolled around a loop](../../../examples/mechanics/rolling_ball/rolling_ball.ipynb), whose orientation is the product of the turns of its steps.
- [Foucault's pendulum](../../../examples/mechanics/foucault/foucault.ipynb), where the part of the earth's turn that lies in the floor's plane turns the swing.
- [Close Kepler passages](../../../examples/mechanics/kepler/kepler.ipynb), where a position is a spinor times its own reverse.
- [Tilings in three signatures](../../../examples/math/reflections/tilings.ipynb), where one construction tiles the sphere, the Euclidean plane and the hyperbolic plane.
- [Koebe polyhedra](../../../examples/surfaces/koebe/koebe.ipynb), where a circle on the sphere is a single vector.

The [index of examples](../../../examples/README.md) lists the rest. The books below treat the topics above in full:

- D. Hestenes, *New Foundations for Classical Mechanics*, 2nd edition, Kluwer, 1999. Geometric algebra developed through mechanics, from the plane to rigid bodies and orbits.
- C. Doran and A. Lasenby, *Geometric Algebra for Physicists*, Cambridge University Press, 2003. The algebra, calculus and their use across physics, from mechanics to quantum theory and gravitation.
- L. Dorst, D. Fontijne and S. Mann, *Geometric Algebra for Computer Science*, Morgan Kaufmann, 2007. The algebra as a tool for geometry, with the conformal model of points, lines, circles and spheres.
- C. Gunn and S. De Keninck, *Geometric Algebra for Computer Graphics*, SIGGRAPH course, 2019, and [bivector.net](https://bivector.net). Projective geometric algebra for points, lines, planes and rigid motions.
- A. Macdonald, *Linear and Geometric Algebra*, 2010. Geometric algebra alongside the linear algebra of a first course.
- D. Hestenes and G. Sobczyk, *Clifford Algebra to Geometric Calculus*, Reidel, 1984. The full mathematical account, including calculus and linear functions of multivectors.
