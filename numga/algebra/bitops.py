from typing import List, Iterable

import numpy as np


# # FIXME: use packbits/unpackbits builtin?
	# r = np.zeros((), dtype=self.elements_dtype)
	# for b in reversed(bits):
	# 	r = np.left_shift(r, 1) + b
	# return r.astype(self.elements_dtype)
def parity_to_sign(p: np.array) -> np.array:
	"""Turn even-odd parity of ints to a sign value
	[0, 1, 2, 3] -> [+1, -1, +1, -1]
	"""
	return (1 - (p % 2) * 2).astype(np.int8)


def minimal_uint_type(bits: int):
	"""Retrieve the minimal integer dtype for holding a given integer value"""
	bytes = (bits - 1) // 8 + 1
	dtype = {1: np.uint8, 2: np.uint16, 3: np.uint32, 4: np.uint32}[bytes]
	return bytes, dtype


def bit_pack(bits: List[bool]) -> np.ndarray:
	"""Packs a sequence of booleans as a singleton integer array"""
	return np.array(int(''.join(str(b) for b in reversed(bits)), 2))


bit_count_radix = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1).astype(np.uint8)
def bit_count(b: np.ndarray, nbytes=None) -> np.ndarray:
	"""Count the bits in each element of array b

	Parameters
	----------
	b: ndarray, [...]

	Returns
	-------
	ndarray, [...], uint8
		bitcounts of each element
	"""
	# print(b.dtype.byteorder)
	bytes = np.ndarray(
		buffer=b.data,
		strides=(1,) + b.strides,   # add new axis that steps over one byte at a time
		# FIXME: not counting all bytes is scary wrt endianness! needs tests!
		# shape=((nbytes or b.dtype.itemsize),) + b.shape,
		shape=(b.dtype.itemsize,) + b.shape,
		dtype=np.uint8          # reinterpret as uint8 so we can do eficient lookup in our radix table
	)
	return sum(bit_count_radix[byte] for byte in bytes)


def biterator(elements: np.array, stop, skip=0) -> Iterable[np.array]:
	"""Slice an array of elements into an iterable of its constituent bits"""
	if skip:
		elements = np.right_shift(elements, skip)
	for i in range(stop - skip):
		yield np.bitwise_and(elements, 1, dtype=np.uint8)
		elements = np.right_shift(elements, 1)
