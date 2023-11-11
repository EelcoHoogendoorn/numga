"""
This module provides 'extension methods' to the multivector type,
that are not simple linear operators generic to all signatures,
but have some reason to be dispatched to different implementations depending on their subspace

This involves mostly
 * norms
 * square roots
 * inverses
 * logarithms
 * exponentials
 * decompositions

Most of these are implemented for dimensions < 6, in a backend and signature agnostic manner.

We tend here to prioritize the most stable and generic algorithms; not the most efficient ones.

"""


from numga.multivector.extension.inverse import *
from numga.multivector.extension.roots import *
from numga.multivector.extension.logexp import *
from numga.multivector.extension.norms import *
