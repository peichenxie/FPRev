from fprev import fprev
from accumimpls.numpy import NumpySum, NumpyDot, NumpyGEMV, NumpyGEMM
from accumimpls.torch import TorchSum, TorchDot, TorchGEMV, TorchGEMM, TorchF16GEMM


def numpy_cpu():
    for accum_impl in [NumpySum, NumpyDot, NumpyGEMV, NumpyGEMM]:
        for n in [4, 5, 6, 7, 8, 16, 32, 64, 128, 256, 512]:
            tree = fprev(accum_impl(n))
            tree.render(f"outputs/{accum_impl.__name__}{n}", format="pdf")


def torch_gpu():
    for accum_impl in [TorchSum, TorchDot, TorchGEMV, TorchGEMM, TorchF16GEMM]:
        for n in [4, 5, 6, 7, 8, 16, 32, 64, 128, 256, 512]:
            tree = fprev(accum_impl(n, use_gpu=True))
            tree.render(f"outputs/{accum_impl.__name__}{n}", format="pdf")


if __name__ == "__main__":
    numpy_cpu()
    torch_gpu()
