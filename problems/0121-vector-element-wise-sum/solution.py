def vector_sum(a: list[int|float], b: list[int|float]) -> list[int|float]:
	if len(a) != len(b):
		return -1
	else:
		s = [0] * len(a)
		for i in range(len(a)):
			s[i] = a[i] + b[i]
		return s