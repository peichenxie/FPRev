import torch
from graphviz import Digraph

from fprev import AccumImpl


class TorchSum(AccumImpl):
    def __init__(self, n: int, use_gpu: bool = False):
        self.n_summands = n
        device = "cuda" if use_gpu else "cpu"
        self.data = torch.ones([n], dtype=torch.float32, device=device)

    def set_mask(self, k: int, negative: bool):
        self.data[k] = -(2.0**127) if negative else 2.0**127

    def reset_mask(self, k: int):
        self.data[k] = 1

    def get_l(self) -> float:
        return self.n_summands - int(self.data.sum().item())

    def random_test(self, tree: Digraph, n_trials: int) -> bool:
        n = self.n_summands
        for _ in range(n_trials):
            A = torch.randn(n)
            sum = A.sum().item()
            order = tree.source.split("\n")
            for line in order:
                if "->" not in line:
                    continue
                line = line.split("->")
                i = int(line[0]) % n
                j = int(line[1]) % n
                if i != j:
                    A[j] += A[i]
            if A[0].item() != sum:
                return False
        return True


class TorchDot(AccumImpl):
    def __init__(self, n: int, use_gpu: bool = False):
        self.n_summands = n
        device = "cuda" if use_gpu else "cpu"
        self.data = torch.ones([n], dtype=torch.float32, device=device)
        self.ones = torch.ones([n], dtype=torch.float32, device=device)

    def set_mask(self, k: int, negative: bool):
        self.data[k] = -(2.0**127) if negative else 2.0**127

    def reset_mask(self, k: int):
        self.data[k] = 1

    def get_l(self) -> float:
        return self.n_summands - int((self.data @ self.ones).item())


class TorchGEMV(AccumImpl):
    def __init__(self, n: int, use_gpu: bool = False):
        self.n_summands = n
        device = "cuda" if use_gpu else "cpu"
        self.data = torch.ones([n], dtype=torch.float32, device=device)
        self.ones = torch.ones([n, n], dtype=torch.float32, device=device)

    def set_mask(self, k: int, negative: bool):
        self.data[k] = -(2.0**127) if negative else 2.0**127

    def reset_mask(self, k: int):
        self.data[k] = 1

    def get_l(self) -> float:
        return self.n_summands - int((self.data @ self.ones)[0].item())


class TorchGEMM(AccumImpl):
    def __init__(self, n: int, use_gpu: bool = False):
        self.n_summands = n
        device = "cuda" if use_gpu else "cpu"
        self.data = torch.ones([n, n], dtype=torch.float32, device=device)
        self.ones = torch.ones([n, n], dtype=torch.float32, device=device)

    def set_mask(self, k: int, negative: bool):
        self.data[0, k] = -(2.0**127) if negative else 2.0**127

    def reset_mask(self, k: int):
        self.data[0, k] = 1

    def get_l(self) -> float:
        return self.n_summands - int((self.data @ self.ones)[0, 0].item())


class TorchF16GEMM(AccumImpl):
    def __init__(self, n: int, use_gpu: bool = False):
        if not use_gpu:
            raise ValueError
        self.n_summands = n
        self.data = torch.full([n, n], 2.0**-24, dtype=torch.float16, device="cuda")
        self.ones = torch.ones([n, n], dtype=torch.float16, device="cuda")

    def set_mask(self, k: int, negative: bool):
        self.data[0, k] = -(2.0**15) if negative else 2.0**15

    def reset_mask(self, k: int):
        self.data[0, k] = 2.0**-24

    def get_l(self) -> float:
        return self.n_summands - int((self.data @ self.ones)[0, 0].item() * 2.0**24)
