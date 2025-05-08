import numpy
from graphviz import Digraph

from fprev import AccumImpl


class NumpySum(AccumImpl):
    def __init__(self, n: int):
        self.n_summands = n
        self.data = numpy.ones([n], dtype=numpy.float32)

    def set_mask(self, k: int, negative: bool):
        self.data[k] = -(2.0**127) if negative else 2.0**127

    def reset_mask(self, k: int):
        self.data[k] = 1

    def get_l(self) -> int:
        return self.n_summands - int(self.data.sum())

    def random_test(self, tree: Digraph, n_trials: int) -> bool:
        n = self.n_summands
        for _ in range(n_trials):
            A = numpy.random.randn(n).astype(numpy.float32)
            sum = A.sum()
            order = tree.source.split("\n")
            for line in order:
                if "->" not in line:
                    continue
                line = line.split("->")
                i = int(line[0]) % n
                j = int(line[1]) % n
                if i != j:
                    A[j] += A[i]
            if A[0] != sum:
                return False
        return True


class NumpyDot(AccumImpl):
    def __init__(self, n: int):
        self.n_summands = n
        self.data = numpy.ones([n], dtype=numpy.float32)
        self.ones = numpy.ones([n], dtype=numpy.float32)

    def set_mask(self, k: int, negative: bool):
        self.data[k] = -(2.0**127) if negative else 2.0**127

    def reset_mask(self, k: int):
        self.data[k] = 1

    def get_l(self) -> int:
        return self.n_summands - int(self.data @ self.ones)


class NumpyGEMV(AccumImpl):
    def __init__(self, n: int):
        self.n_summands = n
        self.data = numpy.ones([n], dtype=numpy.float32)
        self.ones = numpy.ones([n, n], dtype=numpy.float32)

    def set_mask(self, k: int, negative: bool):
        self.data[k] = -(2.0**127) if negative else 2.0**127

    def reset_mask(self, k: int):
        self.data[k] = 1

    def get_l(self) -> int:
        return self.n_summands - int((self.data @ self.ones)[0])


class NumpyGEMM(AccumImpl):
    def __init__(self, n: int):
        self.n_summands = n
        self.data = numpy.ones([n, n], dtype=numpy.float32)
        self.ones = numpy.ones([n, n], dtype=numpy.float32)

    def set_mask(self, k: int, negative: bool):
        self.data[0, k] = -(2.0**127) if negative else 2.0**127

    def reset_mask(self, k: int):
        self.data[0, k] = 1

    def get_l(self) -> int:
        return self.n_summands - int((self.data @ self.ones)[0, 0])
