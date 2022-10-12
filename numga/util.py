"""Some assorted utility function"""

chars = 'abcdefghijklmnopqrstuvwxyz'

try:
	from functools import cache
	from functools import cached_property
except:
	# python < 3.9 compatibility
	from functools import lru_cache as _cache
	from functools import cached_property
	# override stupid default
	def cache(*args, **kwargs):
		return _cache(*args, **{**kwargs, 'maxsize': None})


def match(args):
	"""Return unique value in sequence, if it exists"""
	f = args[0]
	if all(a == f for a in args):
		return f
	else:
		raise Exception(f'Not all elements are equal: {args}')


def reduce(s, op):
	"""Sum over sequence, starting from first element, rather than zero"""
	from functools import reduce
	return reduce(op, s)


def summation(s):
	"""Sum over sequence, starting from first element, rather than zero"""
	import operator
	from functools import reduce
	return reduce(operator.add, s)


def product(s):
	"""Product over sequence, starting from first element, rather than zero"""
	import operator
	from functools import reduce
	return reduce(operator.mul, s)
