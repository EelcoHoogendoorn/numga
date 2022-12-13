Plain python backend.

May seem kinda silly since we have a numpy dependency in this project anyway,
but there could be a legitimate use case for it; if acting on unbatched multivectors,
in a high dimenional space, a sparse product of python scalars should beat the numpy implementation,
since the latter will be dominated by numpy call overhead.

However, its primary purpose is as a benchmarking reference point.
