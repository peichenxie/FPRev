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
