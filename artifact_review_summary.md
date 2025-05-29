# FPRev: Revealing Floating-Point Accumulation Orders - Detailed Artifact Review Summary

## Abstract Summary
FPRev is a diagnostic tool designed to address a critical issue in numerical computing: the lack of documentation regarding accumulation orders in floating-point operations. The tool uses numerical testing to non-intrusively reveal how accumulation-based operations (AccumOps) like summation and matrix multiplication are implemented across different software libraries and hardware platforms. This capability enables developers to ensure numerical reproducibility and verify implementation equivalence across systems.

## Problem Context and Motivation
The core problem stems from the non-associative nature of floating-point arithmetic. For example, computing `(0.5 + 512) + 512.5 = 1025` versus `0.5 + (512 + 512.5) = 1024` in half-precision arithmetic demonstrates how different accumulation orders yield different results. This inconsistency becomes critical in domains requiring numerical reproducibility, such as finance, where even minor discrepancies are unacceptable.

Accumulation-based operations are fundamental to numerous computational domains, yet their accumulation orders are often undocumented in existing software and hardware implementations. This makes it difficult for developers to ensure consistent results across systems, especially with the rapid evolution of heterogeneous hardware and diverse software stacks.

## Technical Approach and Innovation

### 1. Core Algorithm Design
The artifact implements three increasingly sophisticated algorithms:

- **NaiveSol**: A brute-force approach with exponential complexity O(4^n/n^3/2 · t(n))
- **BasicFPRev**: An improved algorithm with quadratic complexity Θ(n²t(n))
- **FPRev**: The optimized solution with complexity between Ω(nt(n)) and O(n²t(n))

### 2. Testing Methodology
The tool uses specially designed numerical inputs to distinguish between different accumulation orders:
- Sets most summands to 1.0
- Places a large number M and its negative -M at strategic positions where M satisfies (n-2) + M = M
- Analyzes the output to determine when cancellation occurs during accumulation
- The summation output corresponds to an integer between 0 and n-2, depending on when M and -M cancel each other
- Constructs summation trees representing the accumulation order using lowest common ancestor (LCA) analysis

### 3. Advanced Hardware Support
FPRev extends beyond basic CPU operations to handle modern GPU matrix accelerators like NVIDIA Tensor Cores, which perform multi-term fused summations. These are modeled using multiway trees rather than binary trees, where summands are aligned and truncated before accumulation, similar to finite-precision fixed-point arithmetic.

## Artifact Structure and Implementation

The codebase is well-organized with the following key components:

### Core Implementation (`fprev.py`)
- **`AccumImpl` class**: Abstract interface for different accumulation implementations
- **`fprev()` function**: Main optimized algorithm for revealing accumulation orders
- **`basic_fprev()` function**: Baseline quadratic algorithm using disjoint sets
- **`naive_sol()` function**: Brute-force reference implementation with exhaustive search

### Library Implementations (`experiments/accumimpls/`)
- **NumPy implementations**: `NumpySum`, `NumpyDot`, `NumpyGEMV`, `NumpyGEMM`
- **PyTorch implementations**: `TorchSum`, `TorchDot`, `TorchGEMV`, `TorchGEMM`, `TorchF16GEMM`
- **JAX implementations**: Various JAX-based accumulation operations

Each implementation follows the `AccumImpl` interface with methods:
- `set_mask()`: Sets large values for testing
- `reset_mask()`: Resets values to 1.0
- `get_l()`: Returns the number of summands accumulated after cancellation
- `random_test()`: Validates the revealed tree against random inputs

### Experimental Framework (`experiments/`)
- **Case Study (`casestudy.py`)**: Analyzes popular libraries on different hardware
- **Efficiency Evaluation**: Three research questions (RQ1-RQ3) examining performance across:
  - **RQ1**: Different libraries (NumPy, PyTorch, JAX)
  - **RQ2**: Different operations (summation, dot product, matrix operations)
  - **RQ3**: Different hardware platforms (CPUs and GPUs)

## Key Findings and Contributions

### 1. Reproducibility Analysis
- **NumPy**: Summation functions show consistent behavior across different CPUs
- **PyTorch**: Summation functions maintain consistency across different GPUs
- **BLAS Dependencies**: Operations relying on Intel MKL, OpenBLAS, and NVIDIA cuBLAS exhibit non-reproducible behavior
- **Tensor Cores**: Special handling required for NVIDIA's specialized matrix accelerators

### 2. Visualization and Documentation
- Generates summation trees showing exactly how operations are performed
- **Example**: NumPy's 32-element summation uses 8-way accumulation optimized for SIMD instructions
- Outputs PDF visualizations using Graphviz for easy interpretation
- Trees show the hierarchical structure of accumulation operations

### 3. Performance Validation
- Demonstrates significant speedup over naive approaches
- Scales effectively across different problem sizes
- Works across diverse hardware architectures (Intel Xeon, AMD EPYC, NVIDIA V100/A100/H100)
- Time complexity improvements make the tool practical for real-world use

## Technical Requirements and Reproducibility

The artifact is designed for comprehensive evaluation:

### Platform Requirements
- **Hardware**: NVIDIA V100 or newer GPUs, various CPU architectures
- **OS**: Ubuntu 22.04
- **Software**: Python 3.11

### Specific Hardware for Paper Reproduction
- CPU: Intel Xeon E5-2690 v4 (24 v-cores), GPU: NVIDIA V100 (5120 CUDA cores)
- CPU: AMD EPYC 7V13 (24 v-cores), GPU: NVIDIA A100 (6912 CUDA cores)
- CPU: Intel Xeon Silver 4210 (40 v-cores), GPU: NVIDIA H100 (16896 CUDA cores)

### Dependencies
- NumPy 1.26.*
- PyTorch 2.3.*
- JAX 0.4.*
- Pandas (for data analysis)
- Graphviz (for visualization)

### Installation and Usage
```bash
sudo apt install graphviz
git clone https://github.com/peichenxie/FPRev.git
cd FPRev
pip install .
pip install -r experiments/requirements.txt
```

## Impact and Applications

FPRev addresses a fundamental challenge in numerical computing by providing:

1. **Transparency**: Reveals undocumented implementation details in popular libraries
2. **Reproducibility**: Enables consistent results across different systems and platforms
3. **Verification**: Allows comparison of implementation equivalence between systems
4. **Optimization Guidance**: Shows how to replicate efficient accumulation patterns
5. **Debugging Support**: Helps identify sources of numerical inconsistencies

### Real-World Applications
- **Financial Computing**: Ensuring consistent results in risk calculations
- **Scientific Computing**: Reproducible simulations across different clusters
- **Machine Learning**: Consistent training results across different hardware
- **Hardware Verification**: Validating accelerator implementations

## Artifact Quality Assessment

The artifact demonstrates excellent software engineering practices:

### Strengths
- **Clean Architecture**: Well-separated concerns with abstract interfaces
- **Comprehensive Testing**: Multiple algorithms for validation and comparison
- **Extensive Evaluation**: Tests across multiple libraries and hardware platforms
- **Good Documentation**: Clear README with detailed reproduction instructions
- **Visualization**: Graphical outputs for intuitive understanding of accumulation orders
- **Modular Design**: Easy to extend for new libraries and operations
- **Performance Focus**: Multiple algorithm variants showing clear complexity improvements

### Technical Rigor
- **Algorithm Correctness**: Multiple validation approaches including random testing
- **Complexity Analysis**: Theoretical time complexity bounds with experimental validation
- **Hardware Coverage**: Support for both traditional and specialized computing units
- **Cross-Platform Testing**: Evaluation across different CPU and GPU architectures

### Research Contributions
1. **Design and Development**: First tool to non-intrusively reveal accumulation orders
2. **Empirical Analysis**: Comprehensive study of popular numerical libraries
3. **Algorithmic Innovation**: Novel approach using specially designed inputs and tree construction
4. **Performance Evaluation**: Thorough efficiency analysis across diverse platforms

## Conclusion

This artifact represents a significant contribution to numerical computing research, providing both a practical tool for developers and a framework for understanding the often-hidden details of floating-point computation implementations across modern computing systems. The tool fills a critical gap in ensuring numerical reproducibility, which is increasingly important as computational workloads move across diverse hardware platforms.

The comprehensive evaluation framework, clean implementation, and practical applicability make this a valuable artifact for the research community and practitioners working with numerical computations where reproducibility is essential. 