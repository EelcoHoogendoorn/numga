from functools import cached_property
from typing import Tuple, List

str_to_sign = {'+': +1, '0': 0, '-': -1}
sign_to_str = {-1: '-', 0: '0', +1: '+'}


def parse_description(description: str):
	"""Parse an algebra description in string format

	Parameters
	----------
	description: str

	Returns
	-------
	names : Tuple[str]
	signature: Tuple[Sign]

	Example
	-------
	'x+y+z+w0'
	"""

	def parse(d):
		last = 0
		for i, c in enumerate(d):
			if c in str_to_sign:
				yield d[last:i], str_to_sign[c]
				last = i + 1
		assert last == len(d)

	names, signature = zip(*parse(description))
	return names, signature


class AlgebraDescription:
	"""All the boilerplate bookkeeping which is generic
	to any particular implementation of the algebra's logic"""
	# FIXME: does this split into description helper class add anything at all? not sure
	#  basically, this description class contains just boring implementation-aspecific helper classes

	# FIXME: should we support non-diagonal metrics? think im fine without them for now
	#  but would be good if we dont code ourselves into a corner that demands a complete rewrite...
	#  hard part about nondiag metric seems to me that each blade product may return not one single blade,
	#  but a linear combination of them

	signature: Tuple[int]
	# FIXME: generators rather than basis?
	basis_names: Tuple[str]

	def __init__(self, basis_names: Tuple[str], signature: Tuple[int]):
		self.basis_names = tuple(basis_names)
		self.signature = tuple(signature)

	@staticmethod
	def from_str(description) -> "AlgebraDescription":
		"""Construct algebra using interleaved basis names and signatures syntax

		Examples
		--------
		'x+y+z+w0' constructs the algebra corresponding to 3d PGA
		"""
		basis_names, signature = parse_description(description)
		return AlgebraDescription(basis_names, signature)

	@staticmethod
	def from_pqr(p: int, q: int, r: int) -> "AlgebraDescription":
		"""Construct algebra using pqr description

		Examples
		--------
		(3, 0, 1) constructs the algebra corresponding to 3d PGA
		"""
		sig = '+' * p + '-' * q + '0' * r
		return AlgebraDescription.from_signature(sig)

	@staticmethod
	def from_signature(sig) -> "AlgebraDescription":
		"""Construct algebra using signature description

		Examples
		--------
		'+++0' constructs the algebra corresponding to 3d PGA
		"""
		from numga.util import chars
		return AlgebraDescription.from_str(''.join(c + s for c, s in zip(chars, sig)))

	@property
	def n_dimensions(self) -> int:
		"""Number of distinct canonical 1-blades in the algebra"""
		return len(self.signature)
	@property
	def n_blades(self) -> int:
		"""Number of distinct k-blades in the algebra"""
		return 2 ** self.n_dimensions
	@property
	def n_grades(self) -> int:
		"""Number of distinct grades present in the algebra"""
		return self.n_dimensions + 1

	@cached_property
	def description_str(self) -> str:
		return ''.join(b+s for b, s in zip(self.basis_names, self.signature_str))
	@cached_property
	def pqr_str(self) -> (int, int, int):
		return '{0}{1}{2}'.format(*self.pqr)
	@cached_property
	def signature_str(self) -> str:
		return ''.join(sign_to_str[s] for s in self.signature)
	@cached_property
	def pqr(self) -> (int, int, int):
		return (sum(self.positives), sum(self.negatives), sum(self.zeros))

	def parse_tokens(self, blade_name: str) -> Tuple[int]:
		"""Given a string like 'yx' parse it into recognized basis tokens"""
		def parse(d: str) -> int:
			last = 0
			for i, c in enumerate(d):
				substr = blade_name[last:i+1]
				if substr in self.basis_names:
					yield self.basis_names.index(substr)
					last = i + 1
			if last < len(d):
				raise Exception(f'Unrecognized blade name {substr}')
		return tuple(parse(blade_name))

	@property
	def zeros(self) -> List[bool]:
		"""0/1 mask indicating all degenerate basis vectors"""
		return [1 if s == 0 else 0 for s in self.signature]
	@property
	def negatives(self) -> List[bool]:
		"""0/1 mask indicating all negative basis vectors"""
		return [1 if s == -1 else 0 for s in self.signature]
	@property
	def positives(self) -> List[bool]:
		"""0/1 mask indicating all positive basis vectors"""
		return [1 if s == +1 else 0 for s in self.signature]
	@property
	def all(self) -> List[bool]:
		"""0/1 mask indicating all basis vectors"""
		return [1 for s in self.signature]

	def __mul__(self, other) -> "AlgebraDescription":
		"""Construct product algebra"""
		assert not set(self.basis_names).intersection(other.basis_names)
		return AlgebraDescription.from_str(self.description_str + other.description_str)
