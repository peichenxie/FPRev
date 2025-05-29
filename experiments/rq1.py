"""
FPRev Performance Evaluation - Research Question 1 (RQ1)
How efficient is FPRev when applied to different libraries?

This script implements the performance evaluation described in Section 7.2 of the FPRev paper,
comparing the efficiency of different accumulation order revelation algorithms across multiple
numerical computing libraries.

Research Question:
-----------------
RQ1 investigates whether FPRev maintains consistent performance characteristics when applied
to different numerical libraries (NumPy, PyTorch, JAX), demonstrating the tool's generalizability
and practical applicability across diverse software ecosystems.

Tested Algorithms:
-----------------
1. NaiveSol (naive_sol): 
   - Brute-force exhaustive search approach
   - Time complexity: O(4^n/n^3/2 · t(n))
   - Used as baseline for comparison, limited to small problem sizes

2. BasicFPRev (basic_fprev):
   - Improved algorithm using disjoint sets and bottom-up tree construction
   - Time complexity: Θ(n²t(n))
   - More practical than naive approach but still quadratic

3. FPRev (fprev):
   - Optimized algorithm with redundancy elimination
   - Time complexity: Ω(nt(n)) to O(n²t(n))
   - Most efficient implementation, suitable for production use

Tested Libraries:
----------------
- NumPy: CPU-based numerical computations using NumPy's sum() function
- PyTorch: GPU-accelerated computations using PyTorch's sum() function  
- JAX: JIT-compiled XLA backend using JAX's sum() function

Each library represents different computational paradigms and backend optimizations,
allowing evaluation of FPRev's performance across diverse execution environments.

Experimental Design:
-------------------
- Adaptive problem size scaling: starts from n=4, increases incrementally or exponentially
- 10 timing runs per configuration for statistical reliability
- Automatic termination when execution time exceeds 1 second or naive solver reaches n=8
- Mean execution time calculation across multiple runs

Performance Metrics:
-------------------
- Execution time (seconds) for each algorithm-library combination
- Scalability analysis showing how performance changes with problem size
- Comparative analysis demonstrating FPRev's efficiency improvements

Expected Results:
----------------
- FPRev should demonstrate significant speedup over BasicFPRev and NaiveSol
- Performance should scale better than quadratic for the optimized FPRev algorithm
- Results should be consistent across different libraries, showing generalizability

Output:
-------
- CSV file: outputs/rq1.csv containing detailed timing results
- Multi-index DataFrame with libraries as columns and problem sizes as rows
- Corresponds to Figure 5 in the paper (when run on consistent hardware)

Hardware Dependencies:
---------------------
- Results may vary based on CPU architecture for NumPy
- GPU performance depends on CUDA capability for PyTorch
- JAX performance depends on XLA compilation and backend optimization

Usage:
------
$ python experiments/rq1.py

The script will automatically scale problem sizes and generate comprehensive timing data
for statistical analysis and performance comparison visualization.
"""

from timeit import default_timer

import pandas

from fprev import AccumImpl, fprev, basic_fprev, naive_sol
from accumimpls.numpy import NumpySum
from accumimpls.torch import TorchSum
from accumimpls.jax import JaxSum


def run(sol, sum_impl: AccumImpl) -> list[float]:
    print(sol.__name__, sum_impl.__class__, sum_impl.n_summands)
    sol(sum_impl)
    times = []
    for t in range(10):
        print(f"Run {t}: ", end="")
        time = default_timer()
        sol(sum_impl)
        time = default_timer() - time
        print(f"{time} sec")
        times.append(time)
    return times


def rq1():
    sols = [naive_sol, basic_fprev, fprev]
    sum_impls = [NumpySum, TorchSum, JaxSum]
    table = []
    n_summands = []
    for sum_impl in sum_impls:
        for sol in sols:
            n = 4
            tested_n = []
            execution_times = []
            while True:
                times = run(sol, sum_impl(n))
                time = sum(times) / 10
                print("mean:", time)
                tested_n.append(n)
                execution_times.append(time)
                if time > 1 or (sol is naive_sol and n == 8):
                    break
                if n < 8:
                    n += 1
                else:
                    n *= 2
            table.append(execution_times)
            if len(tested_n) > len(n_summands):
                n_summands = tested_n
    df = pandas.DataFrame(table).transpose()
    df.index = n_summands
    df.columns = pandas.MultiIndex.from_product(
        [["NumPy", "PyTorch", "JAX"], ["NaiveSol", "BasicFPRev", "FPRev"]]
    )
    print(df)
    df.to_csv("outputs/rq1.csv")


if __name__ == "__main__":
    rq1()
