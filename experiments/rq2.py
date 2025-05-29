"""
FPRev Performance Evaluation - Research Question 2 (RQ2)
How efficient is FPRev when applied to different operations?

This script implements the performance evaluation described in Section 7.3 of the FPRev paper,
analyzing FPRev's efficiency across different types of accumulation-based operations to determine
whether the algorithm's performance characteristics remain consistent regardless of the underlying
mathematical operation being analyzed.

Research Question:
-----------------
RQ2 investigates whether FPRev's performance is dependent on the specific type of accumulation
operation being analyzed, or if the algorithm maintains consistent efficiency across diverse
mathematical operations with different computational complexities and memory access patterns.

Tested Operations:
-----------------
1. Dot Product (NumpyDot):
   - Vector dot product: a·b = Σ(a[i] * b[i])
   - Simple pairwise multiplication followed by summation
   - Linear memory access pattern, cache-friendly
   - Represents basic accumulation pattern

2. General Matrix-Vector Multiplication (NumpyGEMV):
   - Matrix-vector product: y = A*x where A is matrix, x is vector
   - Each output element requires dot product of matrix row with vector
   - More complex memory access patterns (row-wise traversal)
   - Intermediate complexity accumulation operation

3. General Matrix-Matrix Multiplication (NumpyGEMM):
   - Matrix-matrix product: C = A*B where A, B, C are matrices
   - Each output element requires dot product of row and column
   - Most complex operation with nested accumulation loops
   - Highly optimized in practice using BLAS implementations
   - Represents most complex accumulation pattern

Tested Algorithms:
-----------------
Note: NaiveSol is excluded from this evaluation due to its exponential complexity,
which becomes impractical for the larger problem sizes typically used in linear algebra operations.

1. BasicFPRev (basic_fprev):
   - Baseline quadratic algorithm: Θ(n²t(n))
   - Uses disjoint sets for tree construction
   - Serves as comparison baseline for optimization

2. FPRev (fprev):
   - Optimized algorithm with redundancy elimination
   - Time complexity: Ω(nt(n)) to O(n²t(n))
   - Target algorithm for efficiency evaluation

Experimental Design:
-------------------
- Problem size scaling: exponential growth starting from n=4 (n *= 2)
- 10 timing runs per configuration for statistical significance
- Automatic termination when execution time exceeds 1 second
- Focus on practical problem sizes relevant to linear algebra applications

Performance Analysis:
--------------------
The evaluation examines whether:
- FPRev's optimization benefits are consistent across operation types
- More complex operations (GEMM) benefit proportionally from optimization
- Memory access patterns affect the relative performance improvements
- BLAS-optimized operations introduce different performance characteristics

Expected Results:
----------------
- FPRev should consistently outperform BasicFPRev across all operation types
- Performance improvements should be relatively consistent regardless of operation complexity
- More complex operations might show larger absolute time differences
- Results demonstrate FPRev's general applicability to diverse accumulation patterns

Output:
-------
- CSV file: outputs/rq2.csv containing detailed timing results
- Multi-index DataFrame with operations as columns and problem sizes as rows
- Corresponds to Figure 6 in the paper (when run on consistent hardware)

Implementation Notes:
--------------------
- Uses NumPy implementations to ensure consistent backend (typically Intel MKL or OpenBLAS)
- Matrix operations test larger accumulation trees than simple summation
- Results may vary based on underlying BLAS library optimizations

Usage:
------
$ python experiments/rq2.py

The script will automatically scale problem sizes and generate timing data showing
how FPRev performs across different types of accumulation-based operations.
"""

from timeit import default_timer

import pandas

from fprev import AccumImpl, fprev, basic_fprev
from accumimpls.numpy import NumpyDot, NumpyGEMV, NumpyGEMM


def run(sol, accum_impl: AccumImpl) -> list[float]:
    print(sol.__name__, accum_impl.__class__, accum_impl.n_summands)
    sol(accum_impl)
    times = []
    for t in range(10):
        print(f"Run {t}: ", end="")
        time = default_timer()
        sol(accum_impl)
        time = default_timer() - time
        print(f"{time} sec")
        times.append(time)
    return times


def rq2():
    sols = [basic_fprev, fprev]
    accum_impls = [NumpyDot, NumpyGEMV, NumpyGEMM]
    table = []
    n_summands = []
    for accum_impl in accum_impls:
        for sol in sols:
            n = 4
            tested_n = []
            execution_times = []
            while True:
                times = run(sol, accum_impl(n))
                time = sum(times) / 10
                print("mean:", time)
                tested_n.append(n)
                execution_times.append(time)
                if time > 1:
                    break
                n *= 2
            table.append(execution_times)
            if len(tested_n) > len(n_summands):
                n_summands = tested_n
    df = pandas.DataFrame(table).transpose()
    df.index = n_summands
    df.columns = pandas.MultiIndex.from_product(
        [["Dot", "GEMV", "GEMM"], ["BasicFPRev", "FPRev"]]
    )
    print(df)
    df.to_csv("outputs/rq2.csv")


if __name__ == "__main__":
    rq2()
