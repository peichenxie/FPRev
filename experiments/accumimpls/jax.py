import numpy
import jax
from graphviz import Digraph

from fprev import AccumImpl


class JaxSum(AccumImpl):
    def __init__(self, n: int, use_gpu: bool = False):
        self.n_summands = n
        self.device = jax.devices("gpu")[0] if use_gpu else jax.devices("cpu")[0]
        self.data = numpy.ones([n], dtype=numpy.float32)

    def set_mask(self, k: int, negative: bool):
        self.data[k] = -(2.0**127) if negative else 2.0**127

    def reset_mask(self, k: int):
        self.data[k] = 1

    def get_l(self):
        with jax.default_device(self.device):
            return self.n_summands - int(jax.numpy.sum(self.data).item())

    def random_test(self, tree: Digraph, n_trials: int) -> bool:
        n = self.n_summands
        for _ in range(n_trials):
            A = numpy.random.randn(n).astype(numpy.float32)
            sum = jax.numpy.sum(A).item()
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


class JaxDot(AccumImpl):
    def __init__(self, n: int, use_gpu: bool = False):
        self.n_summands = n
        self.device = jax.devices("gpu")[0] if use_gpu else jax.devices("cpu")[0]
        self.data = numpy.ones([n], dtype=numpy.float32)
        self.ones = jax.device_put(
            jax.numpy.ones([n], dtype=jax.numpy.float32), device=self.device
        )

    def set_mask(self, k: int, negative: bool):
        self.data[k] = -(2.0**127) if negative else 2.0**127

    def reset_mask(self, k: int):
        self.data[k] = 1

    def get_l(self):
        with jax.default_device(self.device):
            return self.n_summands - int(jax.numpy.dot(self.data, self.ones).item())


class JaxGEMV(AccumImpl):
    def __init__(self, n: int, use_gpu: bool = False):
        self.n_summands = n
        self.device = jax.devices("gpu")[0] if use_gpu else jax.devices("cpu")[0]
        self.data = numpy.ones([n], dtype=numpy.float32)
        self.ones = jax.device_put(
            jax.numpy.ones([n, n], dtype=jax.numpy.float32), device=self.device
        )

    def set_mask(self, k: int, negative: bool):
        self.data[k] = -(2.0**127) if negative else 2.0**127

    def reset_mask(self, k: int):
        self.data[k] = 1

    def get_l(self):
        with jax.default_device(self.device):
            return self.n_summands - int(jax.numpy.dot(self.data, self.ones)[0].item())


class JaxGEMM(AccumImpl):
    def __init__(self, n: int, use_gpu: bool = False):
        self.n_summands = n
        self.device = jax.devices("gpu")[0] if use_gpu else jax.devices("cpu")[0]
        self.data = numpy.ones([n, n], dtype=numpy.float32)
        self.ones = jax.device_put(
            jax.numpy.ones([n, n], dtype=jax.numpy.float32), device=self.device
        )

    def set_mask(self, k: int, negative: bool):
        self.data[0, k] = -(2.0**127) if negative else 2.0**127

    def reset_mask(self, k: int):
        self.data[0, k] = 1

    def get_l(self):
        with jax.default_device(self.device):
            return self.n_summands - int(
                jax.numpy.dot(self.data, self.ones)[0, 0].item()
            )
