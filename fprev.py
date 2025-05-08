from graphviz import Digraph


class AccumImpl:
    n_summands: int

    def __init__(self): ...
    def set_mask(self, k: int, negative: bool = False): ...
    def reset_mask(self, k: int): ...
    def get_l(self) -> int: ...
    def random_test(self, T: Digraph, n_trials: int) -> bool: ...


def fprev(accum_impl: AccumImpl) -> Digraph:
    T = Digraph()

    def build_subtree(indexes: list[int]) -> tuple[int, int]:
        i = indexes[0]
        T.node(str(i), "#" + str(i))

        if len(indexes) == 1:
            return i, 1

        # calculate L_i
        L_i = []
        accum_impl.set_mask(i, negative=False)
        for j in indexes[1:]:
            accum_impl.set_mask(j, negative=True)
            L_i.append(accum_impl.get_l())
            accum_impl.reset_mask(j)
        accum_impl.reset_mask(i)
        zipped_Li = sorted(zip(L_i, indexes[1:]))
        indexes = [i for _, i in zipped_Li]
        L_i = [x for x, _ in zipped_Li]

        # build the subtree
        current_root = i
        k = 0
        while k < len(indexes):
            cnt = 1
            while k + cnt < len(indexes) and L_i[k + cnt] == L_i[k]:
                cnt += 1
            # J_l == indexes[k : k + cnt]
            root_Tprime, nleaves_Tc = build_subtree(indexes[k : k + cnt])
            if nleaves_Tc == cnt:
                new_root = current_root + accum_impl.n_summands
                T.node(str(new_root), "+")
                T.edge(str(current_root), str(new_root))
                T.edge(str(root_Tprime), str(new_root))
                current_root = new_root
            else:
                T.edge(str(current_root), str(root_Tprime))
                current_root = root_Tprime
            k += cnt
        return current_root, L_i[-1]

    build_subtree(list(range(accum_impl.n_summands)))
    return T


def basic_fprev(accum_impl: AccumImpl) -> Digraph:
    class DisjointSet:
        def __init__(self, n: int):
            self.ancestor = list(range(n))

        def find_root(self, k: int) -> int:
            f = self.ancestor[k % n]
            if f % n == k % n:
                return f
            f = self.find_root(f)
            self.ancestor[k % n] = f
            return f

        def merge(self, i: int, j: int, root: int):
            self.ancestor[i % n] = root
            self.ancestor[j % n] = root

    # calculate L
    n = accum_impl.n_summands
    L = []
    for i in range(n):
        accum_impl.set_mask(i, negative=False)
        for j in range(i + 1, n):
            accum_impl.set_mask(j, negative=True)
            L.append((accum_impl.get_l(), i, j))
            accum_impl.reset_mask(j)
        accum_impl.reset_mask(i)

    # GenerateTree(L)
    T = Digraph()
    for i in range(n):
        T.node(str(i), "#" + str(i))
    S = DisjointSet(n)
    L = sorted(L)
    for _, i, j in L:
        i = S.find_root(i)
        j = S.find_root(j)
        if i != j:
            k = i + n
            T.node(str(k), "+")
            T.edge(str(i), str(k))
            T.edge(str(j), str(k))
            S.merge(i, j, k)
    return T


def naive_sol(accum_impl: AccumImpl) -> Digraph:
    n = accum_impl.n_summands
    V = list(range(n))

    def search(d: int, T: Digraph) -> Digraph:
        if d == n - 1:
            return T if accum_impl.random_test(T, 500) else None
        for i in range(n):
            for j in range(i + 1, n):
                if V[i] >= 0 and V[j] >= 0:
                    new_T = T.copy()
                    new_T.node(str(V[i] + n), "+")
                    new_T.edge(str(V[i]), str(V[i] + n))
                    new_T.edge(str(V[j]), str(V[i] + n))
                    V[i] += n
                    V[j] = -V[j]
                    new_T = search(d + 1, new_T)
                    if new_T is not None:
                        return new_T
                    V[i] -= n
                    V[j] = -V[j]
        return None

    T = Digraph()
    for i in range(n):
        T.node(str(i), "#" + str(i))
    return search(0, T)
