"""
FPRev Accumulation Implementation Package

This package contains implementations of various accumulation-based operations
across different numerical computing libraries, designed to serve as test subjects
for the FPRev diagnostic tool.

Package Structure:
-----------------
- numpy.py: NumPy-based implementations for CPU testing
- torch.py: PyTorch-based implementations for CPU/GPU testing  
- jax.py: JAX-based implementations with XLA compilation

Each module implements the AccumImpl interface to provide standardized testing
capabilities for FPRev across diverse computational backends and hardware platforms.

Purpose:
--------
These implementations enable FPRev to reveal and compare accumulation orders across:
- Different numerical libraries (NumPy, PyTorch, JAX)
- Different hardware platforms (CPU, GPU, TPU)
- Different optimization backends (BLAS, cuBLAS, XLA)
- Different precision modes (float32, float16)

The implementations serve as the foundation for the experimental evaluation
presented in the FPRev paper, including case studies and performance analysis.
"""
