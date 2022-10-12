
import numpy as np


class SubSpaceInterface:
	"""All functionality that can be implemented with just access to the blades and algebra

	Note: both the subspace and operator class implement this interface;
	that is, we can view operators as being/defining their output subspace;
	or view subspaces as nullary operators



	Do the same for multivectors and concrete operators! make mv-interface;
	so when calling/binding a concrete operator, it will always be with a list of
	multivector / subspace, and it will always return an mv-interface.
	so a multivector is also just a nullary multivectoroperator

	should we even be able to bind concrete ops to concrete ops?
	maybe, but i dont really see an application right now so its a headache we can do without

	should concrete types simply implement this same interface?
	GA-element-interface; might be a good way to put it?

	note it shouldnt be hard to also provide support for sparse evaluation of non-nullary
	multivectors. should be useful even if only one term is sparse
	"""
	@property
	def algebra(self):
		raise NotImplementedError
	@property
	def blades(self):
		return NotImplementedError
	@property
	def n_blades(self):
		return len(self.blades)

	# FIXME: this suffices for subspace; but operator would need more!
	def __hash__(self):
		return hash((id(self.algebra), hash(self.blades.tostring())))
	def __eq__(self, other: "SubSpaceInterface"):
		return (self is other) or ((self.algebra is other.algebra) and np.alltrue(self.blades == other.blades))

	def __contains__(self, other: "SubSpaceInterface"):
		return set(other.blades) in set(self.blades)
