"""
NumPy Accumulation Operation Implementations for FPRev Testing

This module implements various accumulation-based operations using NumPy to enable
FPRev to analyze and reveal the accumulation orders used by NumPy's optimized
implementations across different CPU architectures and BLAS backends.

Purpose:
--------
These implementations serve as test subjects for FPRev, allowing the tool to:
- Reveal undocumented accumulation orders in NumPy operations
- Compare consistency across different CPU architectures
- Analyze how BLAS backends (Intel MKL, OpenBLAS) affect accumulation patterns
- Generate summation trees showing exact computation sequences

Implementation Strategy:
-----------------------
Each class implements the AccumImpl interface with specially designed test inputs:
- Normal operation: All elements set to 1.0 for baseline computation
- Testing mode: Strategic placement of large values (±2^127) to detect cancellation points
- Result analysis: Output differences reveal when large values cancel during accumulation

The key insight is that when M + (-M) = 0, the remaining sum equals the number of
elements accumulated after this cancellation, allowing reconstruction of the accumulation tree.

Tested Operations:
-----------------

1. NumpySum - Basic Array Summation
   - Tests: numpy.ndarray.sum() method
   - Pattern: Direct summation of 1D array elements
   - Reveals: Basic accumulation order, SIMD vectorization patterns
   - Significance: Fundamental operation underlying many numerical computations

2. NumpyDot - Dot Product Operation  
   - Tests: numpy dot product (@ operator)
   - Pattern: Element-wise multiplication followed by summation
   - Reveals: How NumPy combines multiplication and accumulation
   - Significance: Core operation in linear algebra and machine learning

3. NumpyGEMV - General Matrix-Vector Multiplication
   - Tests: Matrix-vector product (A @ x where A is matrix, x is vector)
   - Pattern: Each output element computed as dot product of matrix row with vector
   - Reveals: How NumPy handles more complex accumulation in linear algebra
   - Significance: Common operation in solving linear systems and transformations

4. NumpyGEMM - General Matrix-Matrix Multiplication
   - Tests: Matrix-matrix product (A @ B where both are matrices)
   - Pattern: Each output element computed as dot product of row and column
   - Reveals: Most complex accumulation patterns, BLAS optimization strategies
   - Significance: Heavily optimized operation critical for high-performance computing

Technical Details:
-----------------
- Data Type: All operations use float32 for consistent precision analysis
- Large Values: ±2^127 chosen as large enough that (n-2) + 2^127 = 2^127 in float32
- Memory Layout: Contiguous arrays to ensure predictable memory access patterns
- BLAS Integration: Matrix operations leverage underlying BLAS implementations

Testing Methodology:
--------------------
Each implementation provides:
- set_mask(k, negative): Places ±2^127 at position k for testing
- reset_mask(k): Restores position k to 1.0 for normal computation  
- get_l(): Returns count of elements accumulated after cancellation
- random_test(): Validates revealed tree against random inputs for correctness

Expected Findings:
-----------------
Based on the paper's results:
- NumPy summation shows consistent behavior across different CPUs
- Matrix operations may vary based on underlying BLAS implementation
- Accumulation patterns often optimize for SIMD instructions and cache efficiency
- Some operations show platform-specific optimizations

BLAS Dependencies:
-----------------
Results may vary based on:
- Intel MKL: Highly optimized for Intel processors
- OpenBLAS: Open-source alternative with good cross-platform support
- System BLAS: Basic implementation with minimal optimization
- Architecture-specific optimizations (AVX, AVX-512, etc.)

Usage:
------
These classes are typically used by the experimental scripts:
```python
# Test NumPy summation with 32 elements
sum_impl = NumpySum(32)
tree = fprev(sum_impl)
tree.render("numpy_sum_32", format="pdf")
```
"""

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
