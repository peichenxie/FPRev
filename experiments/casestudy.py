"""
FPRev Case Study: Revealing Accumulation Orders in Popular Libraries

This script implements the case study presented in Section 6 of the FPRev paper, which
demonstrates the tool's capability to reveal floating-point accumulation orders in popular
numerical computing libraries across different hardware platforms.

Purpose:
--------
- Analyze accumulation orders in NumPy (CPU-based) and PyTorch (GPU-based) implementations
- Generate visual summation trees showing how different operations are implemented
- Validate FPRev's functionality across diverse accumulation-based operations

Tested Libraries and Operations:
-------------------------------
NumPy (CPU implementations):
- NumpySum: Basic summation operations using NumPy's sum() function
- NumpyDot: Dot product operations using NumPy's @ operator
- NumpyGEMV: General matrix-vector multiplication operations
- NumpyGEMM: General matrix-matrix multiplication operations

PyTorch (GPU implementations):
- TorchSum: GPU-accelerated summation using PyTorch's sum() function
- TorchDot: GPU dot product operations
- TorchGEMV: GPU matrix-vector multiplication
- TorchGEMM: GPU matrix-matrix multiplication  
- TorchF16GEMM: Half-precision (float16) matrix multiplication with Tensor Core support

Test Configurations:
--------------------
- Problem sizes: [4, 5, 6, 7, 8, 16, 32, 64, 128, 256, 512] elements/dimensions
- Each configuration generates a summation tree revealing the accumulation order
- Trees are saved as PDF files in the outputs/ directory for visual analysis

Expected Outputs:
----------------
- PDF files named "{ImplementationName}{Size}.pdf" in outputs/ directory
- Key outputs mentioned in paper:
  * NumpyGEMV8.pdf corresponds to Figure 3 (NumPy implementation analysis)
  * TorchF16GEMM32.pdf corresponds to Figure 4 (PyTorch Tensor Core analysis)

Research Findings:
-----------------
- NumPy operations show consistent accumulation orders across different CPUs
- PyTorch operations maintain consistency across different GPU architectures
- BLAS-dependent operations may exhibit platform-specific variations
- Tensor Core operations use specialized multi-term fused accumulation patterns

Usage:
------
Run this script on different hardware configurations to reproduce the case study results:
$ python experiments/casestudy.py

Hardware Requirements:
---------------------
- CPU: Any modern CPU for NumPy testing
- GPU: NVIDIA V100 or newer for PyTorch testing (especially for Tensor Core analysis)

Note: Results may vary based on underlying BLAS implementations (Intel MKL, OpenBLAS, cuBLAS)
"""

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
