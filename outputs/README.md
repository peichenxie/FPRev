## Major claims of the paper

1. FPRev is functional to reveal the floating-point accumulation orders in common implementations. This claim is detailed by Section 6 "Case study" in the paper. To verify this claim, run `python experiments/casestudy.py` and check the output files in the `outputs` directory. 

    1.1 `outputs/Numpy*.pdf` represents the revealed accumulation orders for NumPy, as discussed in Section 6.1 "NumPy’s implementation on CPUs". Among them, `outputs/NumpyGEMV8.pdf` corresponds to Figure 3 of the paper, if the CPU models are consistent to those in the paper.

    1.2 `outputs/Torch*.pdf` represents the revealed accumulation orders for PyTorch, as discussed in Section 6.2 "PyTorch’s implementation on GPUs". Among them, `outputs/TorchF16GEMM32.pdf` corresponds to Figure 4 of the paper, if the GPU models are consistent to those in the paper.

2. FPRev is efficient. This claim is detailed by Section 7 "Performance evaluation" in the paper. To verify this claim, run `python experiments/rq1.py`, `python experiments/rq2.py`, and `python experiments/rq3.py`, and check the output files in the `outputs` directory.

    2.1 `outputs/rq1.csv` provides the results of Section 7.2 "RQ1: How efficient is FPRev when applied to different libraries?". It corresponds to Figure 5 if the CPU model is consistenst to that in the paper.

    2.2 `outputs/rq2.csv` provides the results of Section 7.3 "RQ2: How efficient is FPRev when applied to different operations?". It corresponds to Figure 6 if the CPU model is consistenst to that in the paper.

    2.3 `outputs/rq3.csv` provides the results of Section 7.4 "RQ3: How efficient is FPRev on different CPUs and GPUs?". It corresponds to Figure 7 if the CPU and GPU models are consistenst to those in the paper.
