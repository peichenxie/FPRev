"""
FPRev Performance Evaluation - Research Question 3 (RQ3)
How efficient is FPRev on different CPUs and GPUs?

This script implements the performance evaluation described in Section 7.4 of the FPRev paper,
examining FPRev's efficiency and scalability across different hardware platforms to demonstrate
the tool's hardware-agnostic performance characteristics and cross-platform applicability.

Research Question:
-----------------
RQ3 investigates whether FPRev maintains consistent performance improvements across different
hardware architectures, including traditional CPUs and modern GPU accelerators, and whether
the optimization benefits scale appropriately with different computational backends.

Hardware Platforms Tested:
--------------------------
1. CPU Execution:
   - Uses PyTorch's CPU backend for matrix multiplication
   - Leverages optimized BLAS libraries (Intel MKL, OpenBLAS)
   - Traditional sequential and vectorized (SIMD) execution
   - Memory hierarchy optimizations through cache-aware algorithms

2. GPU Execution:
   - Uses PyTorch's CUDA backend for GPU-accelerated computation
   - Leverages NVIDIA cuBLAS and potentially Tensor Cores
   - Massively parallel execution with thousands of threads
   - Different memory hierarchy (global, shared, registers)
   - Hardware-specific optimizations and kernel fusion

Test Operation:
--------------
- PyTorch General Matrix-Matrix Multiplication (TorchGEMM)
- Chosen as representative of complex accumulation operations
- Enables direct comparison between CPU and GPU implementations
- Utilizes highly optimized backends on both platforms

Tested Algorithms:
-----------------
1. BasicFPRev (basic_fprev):
   - Baseline quadratic algorithm: Θ(n²t(n))
   - Platform-agnostic implementation
   - Serves as consistent comparison baseline

2. FPRev (fprev):
   - Optimized algorithm with redundancy elimination
   - Time complexity: Ω(nt(n)) to O(n²t(n))
   - Should show consistent improvement regardless of hardware

Experimental Design:
-------------------
- Problem size scaling: exponential growth starting from n=4 (n *= 2)
- 10 timing runs per configuration for statistical reliability
- Automatic termination when execution time exceeds 1 second
- Direct comparison between CPU and GPU execution of identical operations

Performance Analysis Focus:
---------------------------
The evaluation examines:
- Whether FPRev's optimization benefits are hardware-independent
- How different computational backends affect absolute performance
- Relative speedup consistency across CPU vs GPU platforms
- Scalability characteristics on parallel vs sequential architectures

Expected Results:
----------------
- FPRev should consistently outperform BasicFPRev on both CPU and GPU
- Relative performance improvements should be similar across platforms
- GPU execution may show different absolute timings due to parallelization
- Optimization benefits should scale appropriately with hardware capabilities

Hardware Requirements:
---------------------
To reproduce paper results, use specific hardware configurations:
- Intel Xeon E5-2690 v4 (24 v-cores)
- AMD EPYC 7V13 (24 v-cores)  
- Intel Xeon Silver 4210 (40 v-cores)
- NVIDIA V100 (5120 CUDA cores)
- NVIDIA A100 (6912 CUDA cores)
- NVIDIA H100 (16896 CUDA cores)

Implementation Notes:
--------------------
- Uses PyTorch to ensure consistent API across CPU and GPU
- GPU execution requires CUDA-capable hardware
- CPU performance depends on available BLAS implementation
- Results may vary based on driver versions and hardware generations

Output:
-------
- CSV file: outputs/rq3.csv containing detailed timing results
- Multi-index DataFrame with platforms (CPU/GPU) as columns and problem sizes as rows
- Corresponds to Figure 7 in the paper (when run on consistent hardware)

Usage:
------
$ python experiments/rq3.py

Note: Requires both CPU and GPU capabilities. Script will attempt GPU execution;
ensure CUDA and PyTorch GPU support are properly installed.

The script demonstrates FPRev's hardware-agnostic efficiency improvements
and validates its applicability across diverse computational platforms.
"""

from timeit import default_timer

import pandas

from fprev import AccumImpl, fprev, basic_fprev
from accumimpls.torch import TorchGEMM


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


def rq3():
    sols = [basic_fprev, fprev]
    table = []
    n_summands = []
    for use_gpu in [False, True]:
        for sol in sols:
            n = 4
            tested_n = []
            execution_times = []
            while True:
                times = run(sol, TorchGEMM(n, use_gpu))
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
        [["CPU", "GPU"], ["BasicFPRev", "FPRev"]]
    )
    print(df)
    df.to_csv("outputs/rq3.csv")


if __name__ == "__main__":
    rq3()
