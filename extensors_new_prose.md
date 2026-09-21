# Definition

In mathematical terms, an extensor is a multi-linear map from multivectors to a multivector.

In programming terms, extensors allow one to leave open arguments to an expression, and bind them at a later time.

When doing mathematics on the blackboard, one often switches between expressions involving a specific vector, or expressions over the entire space of vectors. Extensor syntax brings that same flexibility to geometric algebra in code, combining expressivity with efficiency of the underlying code.

# Motivation

Geometric relationships deserve to be first-class objects alongside the objects they relate. Inertia, stiffness, and material responses are maps that we need to construct, combine, transform, and solve with. Extensors make those relationships part of the geometric algebra library, expressed through the same operations as the geometry that defines them.

Another consequence of extensors is to bring geometric algebra and conventional linear algebra into one language. In a scene graph, for example, nested rotations, nonuniform scales, and camera projection can compose into a single map before any mesh vertices are supplied. Rotors describe the rigid motions; extensors accommodate the complete transformation. The full strengths of linear algebraic toolkit become available within the same geometric framework.


# Examples

This document will go over some examples demonstrating the practical utlity of extensors and the particulars of their implemention in numga. As a convention, Capitalized names represent multivector spaces, and lower case names concrete multivectors. For instance, `v ^ V` represents the wedge product of a specific vector with the space of all vectors; the result is an extensor of bivector output type, that is unary (having one open argument). 

