"""Some example optimized extension methods

In order to make the most of low level optimizations,
we often need to write code that is tailored to backend specifics,
and we often need to assume a specific memory layout for our multivectors,
or we might be tempted to compromise numerical accuracy, or differentiability;
or allow for restrictions on the signature of the algebra.

All of the above may or may not be appropriate for your application,
and thus by default these extension are not imported and activated;
currently this module serves more as an example of what is possible,
rather than a fully worked out set of functionality.

References
----------
[1] Normalization, Square Roots, and the Exponential and Logarithmic Maps in Geometric Algebras of Less than 6D
[2] May the Forque Be with You; Dynamics in PGA

"""
import numpy as np

# FIXME: move this to numpy backend module?
from numga.multivector.multivector import AbstractMultiVector as mv
@mv.normalized.register(
	# FIXME: the tie-in with specific axes names isnt really required is it?
	#  s.algebra.description.signature_str == '+++0' should suffice?
	#  it does depend on the internals of the subspace class and how it chooses to order its blades
	#  this method tied-in to basis names doesnt couple to those subspace internal details
	lambda s:
	s.named_str == '1,xy,xz,yz,xw,yw,zw,xyzw' and
	s.algebra.description.description_str == 'x+y+z+w0',
	position=0      # try and match this before all else
)
def normalize_3dpga_motor_numpy(m: "Motor") -> "Motor":
	"""Optimized 3d pga motor normalization, for the given memory layout

	Notes
	-----
	Terms involving xz are negated compared to the reference [1],
	since our normalized blade order is to sort the axes alphabetically,
	and our subspace object does not currently store signs of basis blades

	Note that this implementation is numpy-specific;
	we would need to rewrite the inplace updates for JAX;
	but a more generic method that would work for both
	would add overhead in a non-compiled numpy context,
	which is rather at odds with low level optimization.
	"""
	m = m.copy()
	# slice backing array into views onto components
	e, xy, xz, yz, xw, yw, zw, xyzw = [m.values[..., i:i+1] for i in range(8)]
	s = (e*e + xy*xy + xz*xz + yz*yz) ** (-0.5)
	d = (e*xyzw - xy*zw + xz*yw - yz*xw) * s * s
	m.values *= s; xw += yz * d; yw -= xz * d; zw += xy * d; xyzw -= e * d
	return m

# reset cache
mv.normalized.cache = {}
@mv.exp.register(
	lambda s:
	s.named_str == 'xw,yw,zw' and
	s.algebra.description.description_str == 'x+y+z+w0',
	position=0      # try and match this before all else
)
def exponentiate_degenerate_bivector_numpy(b: "BiVector") -> "Motor":
	"""Optimized 3d pga exponentiation, for the given memory layout
	"""
	M = np.empty(b.shape + (4,))
	M[..., 0] = 1
	M[..., 1:] = b.values
	return b.context.multivector(values=M, subspace=b.context.subspace.translator())

@mv.exp.register(
	lambda s:
	s.named_str == 'xy,xz,yz' and
	s.algebra.description.description_str == 'x+y+z+w0',
	position=0      # try and match this before all else
)
def exponentiate_nondegenerate_bivector_numpy(b: "BiVector") -> "Motor":
	"""Optimized 3d pga exponentiation, for the given memory layout
	"""
	xy, xz, yz = np.moveaxis(b.values, -1, 0)
	l = xy*xy + xz*xz + yz*yz
	a = np.sqrt(l)
	c = np.cos(a)
	s = np.where(a > 1e-20, np.sin(a) / a, 1)

	M = np.empty(b.shape + (4,))
	M[..., 0] = c
	M[..., 1] = s*xy
	M[..., 2] = s*xz
	M[..., 3] = s*yz
	return b.context.multivector(values=M, subspace=b.context.subspace.rotor())

@mv.exp.register(
	lambda s:
	s.named_str == 'xy,xz,yz,xw,yw,zw' and
	s.algebra.description.description_str == 'x+y+z+w0',
	position=0      # try and match this before all else
)
def exponentiate_bivector_numpy(b: "BiVector") -> "Motor":
	"""Optimized 3d pga exponentiation, for the given memory layout

	Notes
	-----
	Terms involving xz are negated compared to the reference [1],
	since our normalized blade order is to sort the axes alphabetically,
	and our subspace object does not currently store signs of basis blades
	"""
	xy, xz, yz, xw, yw, zw = np.moveaxis(b.values, -1, 0)
	l = xy*xy + xz*xz + yz*yz
	a = np.sqrt(l)
	m = xy*zw - xz*yw + yz*xw
	c = np.cos(a)
	s = np.where(a > 1e-20, np.sin(a) / a, 1)
	t = m * (c - s) / l

	M = np.empty(b.shape + (8,))
	M[..., 0] = c
	M[..., 1] = s*xy
	M[..., 2] = s*xz
	M[..., 3] = s*yz
	M[..., 4] = s*xw + t*yz
	M[..., 5] = s*yw - t*xz
	M[..., 6] = s*zw + t*xy
	M[..., 7] = m*s
	return b.context.multivector(values=M, subspace=b.context.subspace.motor())

# reset cache
mv.exp.cache = {}
