# FPRev: Revealing Floating-Point Accumulation Orders in Software/Hardware Implementations

This repository includes the source code of FPRev and the source code for reproducing the experiments of the paper "Revealing Floating-Point Accumulation Orders in Software/Hardware Implementations". Citation:
```
@inproceedings{xie2025revealing,
  author    = {Peichen Xie and Yanjie Gao and Yang Wang and Jilong Xue},
  title     = {Revealing Floating-Point Accumulation Orders in Software/Hardware Implementations},
  booktitle = {Proceedings of the 2025 USENIX Annual Technical Conference (USENIX ATC)},
  year      = {2025},
  publisher = {USENIX Association}
}
```

## How to reproduce the experiments

### Platform requirements

1. GPU: NVIDIA V100 or newer
2. OS: Ubuntu 22.04
3. Software: Python (version 3.11)

If you wish to reproduce the results in the paper, please use the identical CPU and GPU models:

1. CPU: Intel Xeon E5-2690 v4 (24 v-cores), GPU: NVIDIA V100 (5120 CUDA cores)
2. CPU: AMD EPYC 7V13 (24 v-cores), GPU: NVIDIA A100 (6912 CUDA cores)
3. CPU: Intel Xeon Silver 4210 (40 v-cores), GPU: NVIDIA H100 (16896 CUDA cores)

### Installation

```
sudo apt install graphviz
git clone https://github.com/peichenxie/FPRev.git
cd FPRev
pip install .
pip install -r experiments/requirements.txt
```

### Running experiments

- To reproduce the results in Section 6 (Case study), run `python experiments/casestudy.py` on different hardware models.
- To reproduce the results in Section 7.2 (RQ1: How efficient is FPRev when applied to different libraries?), run `python experiments/rq1.py`.
- To reproduce the results in Section 7.3 (RQ2: How efficient is FPRev when applied to different operations?), run `python experiments/rq2.py`.
- To reproduce the results in Section 7.4 (RQ3: How efficient is FPRev on different CPUs and GPUs?), run `python experiments/rq3.py` on different hardware models.

Then, check the output files in the `outputs` directory. See [outputs/README.md](outputs/README.md) for more information.
