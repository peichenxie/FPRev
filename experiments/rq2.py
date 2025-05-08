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
