"""
JAX Accumulation Operation Implementations for FPRev Testing

This module implements various accumulation-based operations using Google's JAX library
to enable FPRev to analyze and reveal the accumulation orders used by JAX's XLA-compiled
implementations across different computational backends and hardware accelerators.

Purpose:
--------
These implementations serve as test subjects for FPRev to analyze:
- XLA-compiled accumulation orders in JAX operations
- Just-In-Time (JIT) compilation effects on accumulation patterns
- Cross-platform consistency across CPU, GPU, and TPU backends
- Automatic differentiation system interaction with accumulation operations

Key Features of JAX:
-------------------
- XLA Compilation: Just-in-time compilation for optimized execution
- Device Agnostic: Unified API across CPU, GPU, and TPU
- Functional Programming: Pure functions with automatic differentiation
- Vectorization: Automatic batching and vectorization transformations
- Backend Flexibility: Multiple execution backends (CPU, CUDA, TPU)

Implementation Strategy:
-----------------------
JAX-specific considerations for FPRev testing:
- Device Management: Explicit device placement using jax.device_put()
- XLA Compilation: Operations compiled to optimized XLA HLO (High-Level Optimizer)
- Memory Layout: XLA may reorganize data for optimal execution
- Fusion Optimization: XLA automatically fuses operations for efficiency
- Backend Abstraction: Unified interface across different hardware

Tested Operations:
-----------------

1. JaxSum - XLA-Compiled Summation
   - Tests: jax.numpy.sum() with XLA compilation
   - XLA execution: Optimized reduction kernels across backends
   - Reveals: How XLA optimizes summation for target hardware
   - Significance: Foundation for gradient accumulation in automatic differentiation

2. JaxDot - XLA Dot Product Operation
   - Tests: jax.numpy.dot() with XLA optimization
   - XLA execution: Backend-specific optimized kernels (cuBLAS, MKL, etc.)
   - Reveals: XLA's approach to vectorized operations
   - Significance: Core building block for neural network computations in JAX

3. JaxGEMV - XLA Matrix-Vector Multiplication
   - Tests: Matrix-vector product with JAX arrays
   - XLA execution: Optimized GEMV operations for target backend
   - Reveals: XLA memory layout and computation optimization strategies
   - Significance: Common operation in linear transformations and neural networks

4. JaxGEMM - XLA Matrix-Matrix Multiplication
   - Tests: General matrix multiplication with JAX
   - XLA execution: Highly optimized GEMM implementations per backend
   - Reveals: XLA's sophisticated matrix operation optimization
   - Significance: Critical for efficient neural network training and inference

Technical Specifications:
------------------------
Data Management:
- Input Data: NumPy arrays (numpy.float32) for consistent initialization
- Device Placement: Explicit jax.device_put() for proper device allocation
- Type System: JAX's unified type system across different backends
- Large Values: ±2^127 for masking (consistent with NumPy/PyTorch)

XLA Compilation Pipeline:
- Frontend: JAX NumPy API calls
- Middle-end: XLA HLO optimization passes
- Backend: Platform-specific code generation (LLVM, NVPTX, etc.)
- Runtime: Optimized kernels executed on target hardware

Device Management:
------------------
Device Selection:
- CPU: jax.devices("cpu")[0] for CPU execution
- GPU: jax.devices("gpu")[0] for CUDA/ROCm execution  
- TPU: jax.devices("tpu")[0] for TPU execution (if available)
- Default Context: jax.default_device() for operation execution

Backend Optimization:
- CPU Backend: Leverages optimized BLAS (Intel MKL, OpenBLAS)
- GPU Backend: Uses cuBLAS, cuDNN, and custom CUDA kernels
- TPU Backend: Specialized TPU matrix units and memory hierarchy
- XLA Fusion: Automatic kernel fusion for reduced memory traffic

Expected XLA Optimizations:
--------------------------
Based on XLA's optimization capabilities:
- Operation Fusion: Multiple operations combined into single kernels
- Memory Layout: Data reorganization for optimal access patterns
- Vectorization: Automatic SIMD instruction generation
- Loop Optimization: Unrolling, tiling, and parallelization
- Backend Specialization: Hardware-specific optimization passes

Research Implications:
---------------------
JAX's XLA compilation introduces unique considerations:
- Accumulation orders may vary based on XLA optimization level
- Backend-specific optimizations can affect revealed patterns
- JIT compilation may produce different results than eager execution
- Cross-platform consistency depends on XLA's optimization stability

Performance Characteristics:
---------------------------
- First Run: JIT compilation overhead (higher latency)
- Subsequent Runs: Optimized compiled code (lower latency)
- Memory Usage: XLA may allocate additional buffers for optimization
- Scalability: Automatic parallelization across available hardware

Hardware Requirements:
---------------------
CPU Testing:
- Any modern CPU with XLA CPU backend support
- Performance depends on available BLAS and vectorization support

GPU Testing:
- NVIDIA GPU with CUDA Compute Capability 6.0+
- CUDA toolkit and JAX GPU installation
- Sufficient GPU memory for XLA buffer allocation

TPU Testing (if available):
- Google Cloud TPU or local TPU hardware
- JAX TPU backend installation
- Appropriate TPU driver and runtime

Usage Examples:
--------------
```python
# Test JAX CPU summation with XLA compilation
cpu_sum = JaxSum(64, use_gpu=False)
tree = fprev(cpu_sum)

# Test JAX GPU matrix multiplication
gpu_gemm = JaxGEMM(32, use_gpu=True)
tree = fprev(gpu_gemm)
```

Note: JAX operations are automatically JIT-compiled, so accumulation patterns
may reflect XLA's optimization decisions rather than naive implementation choices.
"""

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
