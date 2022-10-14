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
